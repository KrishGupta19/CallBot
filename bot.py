import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Pipecat core
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Pipecat processors
from pipecat.processors.aggregators.llm_context import LLMContext

# Pipecat services
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService  
from pipecat.services.cartesia.tts import CartesiaTTSService

# Pipecat VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# Pipecat Twilio transport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

# Pipecat frames
from pipecat.frames.frames import (
    LLMContextFrame, 
    EndFrame, 
    TranscriptionFrame, 
    TextFrame, 
    LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

class ConversationTracker:
    """Tracks which qualification questions have been answered."""
    
    def __init__(self):
        self.caller_name = None
        self.interest = None
        self.budget = None
        self.timeline = None
        self.source = None
        self.transcript = []          # List of {"role": "user"/"assistant", "text": "..."}
        self.call_start_time = None
        self.call_end_time = None
        self.caller_number = None
    
    def add_turn(self, role: str, text: str):
        """Add a conversation turn to the transcript."""
        self.transcript.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_transcript_text(self) -> str:
        """Get the full transcript as readable text."""
        lines = []
        for turn in self.transcript:
            label = "Caller" if turn["role"] == "user" else "AI Agent"
            lines.append(f"{label}: {turn['text']}")
        return "\n".join(lines)
    
    def get_duration_seconds(self) -> int:
        """Get call duration in seconds."""
        if self.call_start_time and self.call_end_time:
            return int((self.call_end_time - self.call_start_time).total_seconds())
        return 0


class TranscriptLogger(FrameProcessor):
    """Captures conversation text for transcript logging."""
    
    def __init__(self, tracker: ConversationTracker, role: str):
        super().__init__()
        self.tracker = tracker
        self.role = role
        self._current_text = ""
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame) and self.role == "user":
            self.tracker.add_turn("user", frame.text)
            logger.info(f"Caller said: {frame.text}")
        
        elif isinstance(frame, TextFrame) and self.role == "assistant":
            self._current_text += frame.text
            
        elif isinstance(frame, LLMFullResponseEndFrame) and self.role == "assistant":
            if self._current_text.strip():
                self.tracker.add_turn("assistant", self._current_text.strip())
                logger.info(f"AI Agent said: {self._current_text.strip()}")
                self._current_text = ""
        
        await self.push_frame(frame, direction)

async def save_call_data(tracker: ConversationTracker):
    """Save call transcript and metadata to a JSON file."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent / "call_logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Generate filename from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caller = tracker.caller_number or "unknown"
    filename = f"call_{timestamp}_{caller}.json"
    
    call_data = {
        "caller_number": tracker.caller_number,
        "call_start": tracker.call_start_time.isoformat() if tracker.call_start_time else None,
        "call_end": tracker.call_end_time.isoformat() if tracker.call_end_time else None,
        "duration_seconds": tracker.get_duration_seconds(),
        "transcript": tracker.transcript,
        "transcript_text": tracker.get_transcript_text(),
    }
    
    filepath = logs_dir / filename
    filepath.write_text(json.dumps(call_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Call transcript saved to: {filepath}")
    
    return call_data

async def generate_call_summary(tracker: ConversationTracker) -> dict:
    """Use Groq to generate a structured summary of the call."""
    from groq import AsyncGroq
    try:
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return {"error": str(e)}
        
    transcript_text = tracker.get_transcript_text()
    if not transcript_text:
        return {"summary": "No conversation occurred."}
    
    summary_prompt = f"""Analyze this sales call transcript and extract the following information.
Return ONLY valid JSON with these exact keys. If a field was not discussed, use null.

{{
    "caller_name": "string or null",
    "interest": "what product/service they're interested in, or null",
    "budget_range": "their stated budget, or null",
    "timeline": "when they want to start, or null",
    "source": "how they heard about us, or null",
    "lead_score": "1-10 integer based on how interested they seemed",
    "sentiment": "positive / neutral / negative",
    "summary": "2-3 sentence summary of the call",
    "follow_up_action": "recommended next step"
}}

TRANSCRIPT:
{transcript_text}"""

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        
        summary_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        if summary_text.startswith("```"):
            summary_text = summary_text.split("```")[1]
            if summary_text.startswith("json"):
                summary_text = summary_text[4:]
            summary_text = summary_text.strip()
        
        summary = json.loads(summary_text)
        logger.info(f"Call summary generated.")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return {"error": str(e), "raw_transcript": transcript_text}

async def run_bot(websocket, stream_sid: str, call_sid: str = None):
    """
    Creates and runs the complete voice pipeline for one phone call.
    Called by server.py when a WebSocket connection is established.
    """
    load_dotenv()

    # 1. Set up the Twilio WebSocket transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.7,      # Wait 700ms of silence before deciding speech ended
                )
            ),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                call_sid=call_sid,
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
            ),
        ),
    )

    # 2. Set up Groq STT (Whisper)
    stt = GroqSTTService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="whisper-large-v3-turbo",
    )

    # 3. Set up Groq LLM (Llama 3.3 70B)
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    # 4. Set up Cartesia TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Default English voice
        # Try these voice IDs for different sounds:
        # "a0e99841-438c-4a64-b679-ae501e7d6091" — friendly male
        # "156fb8d2-335b-4950-9cb3-a2d33befec77" — friendly female
        # Check https://play.cartesia.ai for more voices
        sample_rate=8000,  # Must be 8000 for Twilio (phone audio)
    )

    # 5. Set up the conversation context (system prompt + history)
    messages = [
        {
            "role": "system",
            "content": """You are a friendly and professional AI sales assistant for VCare Techs, 
an AI-powered sales and support automation company.

You speak naturally in Hinglish — mixing Hindi and English the way people 
naturally talk in urban India. For example: "Haan, definitely! Aapko kis 
type ki service chahiye?" or "Great, that's a good budget range."

YOUR JOB ON THIS CALL:
1. Greet the caller warmly. Introduce yourself: "Hi! Main VCare Techs se 
   bol raha hoon. Aapka naam kya hai?"
2. Ask what service or product they are interested in.
3. Ask about their approximate budget range.
4. Ask when they want to get started — this month, next month, or later.
5. Ask how they heard about VCare Techs.
6. After collecting all info, thank them and say: "Thank you so much! 
   Hamare team se koi aapko detail mein call karega for a demo. 
   Have a great day!"

RULES:
- Keep EVERY response to 1-2 sentences MAX. This is a phone call, not a lecture.
- Sound warm, natural, and conversational. Not robotic.
- Use natural fillers like "achha", "haan", "okay great", "bilkul" between turns.
- If asked about specific pricing or technical details, say: "Woh details 
  hamare team better explain kar payega — main unhe aapki info forward 
  kar dunga."
- If the caller goes off-topic, gently redirect: "Haan woh interesting hai, 
  but let me focus on aapki requirement first..."
- NEVER make up product names, pricing, or features. Keep it general.
- If the caller says they're not interested, be polite: "No problem at all! 
  Agar future mein kuch chahiye toh feel free to call back. Thank you!"
- End the call gracefully after collecting all information."""
        }
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create tracker for this call
    tracker = ConversationTracker()
    tracker.call_start_time = datetime.now()

    # Create transcript loggers
    user_transcript_logger = TranscriptLogger(tracker, "user")
    assistant_transcript_logger = TranscriptLogger(tracker, "assistant")

    # 6. Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),              # Audio from Twilio
            stt,                            # Speech to text (Groq Whisper)
            user_transcript_logger,         # Log what caller says
            context_aggregator.user(),      # Add user message to context
            llm,                            # Generate response (Groq Llama)
            assistant_transcript_logger,    # Log what assistant says
            tts,                            # Text to speech (Cartesia)
            transport.output(),             # Audio back to Twilio
            context_aggregator.assistant(), # Add assistant message to context
        ]
    )

    # 7. Create and run the task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,       # Enable barge-in
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # 8. Event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Caller connected")
        # Trigger the first message from the AI (greeting)
        await task.queue_frames([LLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected")
        tracker.call_end_time = datetime.now()
        
        # Save transcript
        call_data = await save_call_data(tracker)
        
        # Generate summary
        summary = await generate_call_summary(tracker)
        
        # Save summary to the same file
        logs_dir = Path(__file__).parent / "call_logs"
        try:
            latest_file = max(logs_dir.glob("call_*.json"), key=lambda p: p.stat().st_mtime)
            existing_data = json.loads(latest_file.read_text(encoding="utf-8"))
            existing_data["summary"] = summary
            latest_file.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Summary added to: {latest_file.name}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            
        await task.queue_frames([EndFrame()])

    # 9. Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)
