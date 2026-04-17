import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from groq import Groq

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies, default_user_turn_start_strategies
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

from pipecat.frames.frames import (
    LLMContextFrame,
    EndFrame,
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

load_dotenv()

BOT_PROFILES = {
    "doctor": {
        "label": "Doctor appointments",
        "system_prompt": """You are a clinic receptionist for a doctor.
You speak naturally in Hinglish (Hindi + English).

Goal: book an appointment or mark as needs-callback.

Collect these details by asking short, one-at-a-time questions:
- patient_name
- phone_number (confirm if it matches caller ID; if not, ask)
- is_new_patient (yes/no)
- reason_for_visit (1 sentence)
- preferred_date (ask for day + date)
- preferred_time (morning/afternoon/evening + exact time if possible)
- doctor_preference (if any)

Rules:
- If symptoms sound urgent/emergency, tell them to seek urgent medical care immediately and mark outcome as "emergency_redirect".
- Keep responses 1-2 lines. Be polite and efficient.
""",
        "intake_schema_hint": """Return JSON with:
business_type="doctor"
outcome: one of ["booked","needs_callback","emergency_redirect","unknown"]
patient_name, phone_number, is_new_patient, reason_for_visit, preferred_date, preferred_time, doctor_preference
notes (string)
""",
    },
    "bakery": {
        "label": "Bakery orders",
        "system_prompt": """You are a bakery ordering assistant.
You speak naturally in Hinglish (Hindi + English).

Goal: confirm a bakery order or mark as needs-callback.

Collect these details by asking short, one-at-a-time questions:
- customer_name
- phone_number (confirm if it matches caller ID; if not, ask)
- items (what they want)
- quantity
- pickup_or_delivery ("pickup" or "delivery")
- desired_date
- desired_time
- address (only if delivery)
- special_instructions

Rules:
- Keep responses 1-2 lines. Confirm the order summary before ending.
""",
        "intake_schema_hint": """Return JSON with:
business_type="bakery"
outcome: one of ["order_confirmed","needs_callback","unknown"]
customer_name, phone_number, items, quantity, pickup_or_delivery, desired_date, desired_time, address, special_instructions
notes (string)
""",
    },
}


async def extract_intake(business_type: str | None, transcript_text: str) -> dict | None:
    bt = (business_type or "").strip().lower()
    profile = BOT_PROFILES.get(bt)
    if not profile:
        return None
    if not transcript_text.strip():
        return {"business_type": bt, "outcome": "unknown", "notes": "Empty transcript"}

    prompt = f"""You extract structured intake details from a phone call transcript.
Return ONLY valid JSON. No markdown.

{profile["intake_schema_hint"]}

Transcript:
{transcript_text}
"""

    def _run():
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Use a broadly available Groq model; keep it stable.
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_SUMMARY_MODEL") or "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)

    try:
        import asyncio

        return await asyncio.to_thread(_run)
    except Exception as e:
        logger.warning(f"Intake extraction failed: {e}")
        return None


def compute_lead_score(intake: dict | None) -> int | None:
    """
    Dashboard reads lead score from payload.summary.lead_score.

    Rules (as requested):
    - Doctor: good lead if appointment is booked.
    - Bakery: good lead if cake order is made (confirmed).
    """
    if not isinstance(intake, dict):
        return None

    bt = (intake.get("business_type") or "").strip().lower()
    outcome = (intake.get("outcome") or "").strip().lower()

    if bt == "doctor":
        if outcome == "booked":
            return 10
        if outcome == "needs_callback":
            return 5
        if outcome == "emergency_redirect":
            return 0
        return 1

    if bt == "bakery":
        items = (intake.get("items") or "").strip().lower()
        if outcome == "order_confirmed":
            return 10 if "cake" in items else 7
        if outcome == "needs_callback":
            return 4
        return 1

    return None


# ---------------- TRANSCRIPT LOGGER ---------------- #
class TranscriptLogger(FrameProcessor):
    def __init__(self, role, tracker):
        super().__init__()
        self.role = role
        self.tracker = tracker
        self.buffer = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and self.role == "user":
            logger.info(f"USER: {frame.text}")
            self.tracker.append({"role": "user", "text": frame.text})

        elif isinstance(frame, TextFrame) and self.role == "assistant":
            self.buffer += frame.text

        elif isinstance(frame, LLMFullResponseEndFrame) and self.role == "assistant":
            if self.buffer.strip():
                logger.info(f"AI: {self.buffer}")
                self.tracker.append({"role": "assistant", "text": self.buffer})
                self.buffer = ""

        await self.push_frame(frame, direction)


# ---------------- SAVE ---------------- #
async def save_call(
    tracker,
    *,
    call_start: datetime | None,
    call_end: datetime | None,
    caller_number: str | None,
    call_sid: str | None,
    business_type: str | None,
):
    logs = Path("call_logs")
    logs.mkdir(exist_ok=True)

    now = datetime.now()
    file = logs / f"call_{now.strftime('%H%M%S')}.json"

    start_iso = call_start.isoformat() if call_start else None
    end_iso = call_end.isoformat() if call_end else None
    duration_seconds = None
    if call_start and call_end:
        duration_seconds = int((call_end - call_start).total_seconds())

    transcript = tracker if isinstance(tracker, list) else []
    transcript_text = " ".join([(t.get("text") or "").strip() for t in transcript if isinstance(t, dict)]).strip()

    intake = await extract_intake(business_type, transcript_text)
    lead_score = compute_lead_score(intake)

    payload = {
        "filename": file.name,
        "call_sid": call_sid,
        "caller_number": caller_number,
        "business_type": business_type,
        "call_start": start_iso,
        "call_end": end_iso,
        "duration_seconds": duration_seconds,
        "transcript": transcript,
        "transcript_text": transcript_text,
        "intake": intake,
        "summary": {
            "caller_name": None,
            "interest": None,
            "budget_range": None,
            "timeline": None,
            "source": None,
            "lead_score": lead_score,
            "sentiment": None,
            "summary": None,
            "follow_up_action": None,
        },
    }

    file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {file}")


# ---------------- MAIN BOT ---------------- #
async def run_bot(
    websocket,
    stream_sid: str,
    call_sid: str = None,
    caller_number: str = None,
    business_type: str = None,
):

    # -------- TRANSPORT -------- #
    # NOTE: vad_enabled/vad_analyzer do NOT exist in FastAPIWebsocketParams
    # in pipecat 1.0.0 — VAD must be configured in LLMUserAggregatorParams below.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_sid,
                params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
            ),
        ),
    )

    # -------- SERVICES -------- #
    stt = GroqSTTService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqSTTService.Settings(model="whisper-large-v3-turbo"),
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=60,
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
        sample_rate=8000,  # Must match Twilio's 8000 Hz
    )

    # -------- PROMPT -------- #
    bt = (business_type or "").strip().lower()
    profile = BOT_PROFILES.get(bt) or BOT_PROFILES["doctor"]
    messages = [
        {
            "role": "system",
            "content": profile["system_prompt"],
        }
    ]

    context = LLMContext(messages)

    # VAD lives here in pipecat 1.0.0 — it broadcasts VADUserStartedSpeakingFrame /
    # VADUserStoppedSpeakingFrame upstream so SegmentedSTTService can transcribe audio.
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=8000,
                params=VADParams(stop_secs=1.5),
            ),
            user_turn_strategies=UserTurnStrategies(
                start=default_user_turn_start_strategies(),
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8)],
            ),
        ),
    )

    tracker = []
    user_log = TranscriptLogger("user", tracker)
    ai_log = TranscriptLogger("assistant", tracker)

    call_start_dt: datetime | None = None
    call_end_dt: datetime | None = None

    # -------- PIPELINE -------- #
    # LLMTrigger removed: aggregator.user() handles triggering the LLM
    # natively via SpeechTimeoutUserTurnStopStrategy.
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_log,
        aggregator.user(),
        llm,
        ai_log,
        tts,
        transport.output(),
        aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Match Twilio's 8000 Hz audio
        ),
    )

    # -------- EVENTS -------- #
    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        logger.info("CALL STARTED")
        nonlocal call_start_dt
        call_start_dt = datetime.now()
        await task.queue_frames([LLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        logger.info("CALL ENDED")
        nonlocal call_end_dt
        call_end_dt = datetime.now()
        await save_call(
            tracker,
            call_start=call_start_dt,
            call_end=call_end_dt,
            caller_number=caller_number,
            call_sid=call_sid,
            business_type=bt,
        )
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)
