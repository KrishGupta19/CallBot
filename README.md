# AI Voice Agent — Pipecat + Twilio + Groq

A real-time AI phone agent that answers calls, speaks Hinglish, and qualifies sales leads.

## Prerequisites
- Python 3.11+
- ngrok
- Twilio account
- Groq API key
- Cartesia API key

## Setup

1. Clone the repo
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in all API keys
5. Start ngrok in terminal 1: `ngrok http 8765`
6. Copy the ngrok HTTPS URL and update `.env` NGROK_URL
7. Go to Twilio Console → Phone Numbers → your number → Voice Configuration
8. Set "When a call comes in" webhook to: `https://YOUR-NGROK-URL/incoming-call` (HTTP POST)
9. Start the server in terminal 2: `python server.py`
10. Call your Twilio number from any phone

## Troubleshooting

- **Call connects but no audio** → Check ngrok URL matches what's in `.env` and Twilio
- **AI doesn't respond** → Check Groq API key is valid
- **Voice sounds choppy** → Check internet connection, try reducing VAD `stop_secs`
- **AI cuts me off while I'm talking** → Increase `stop_secs` in `VADParams` to 0.8 or 1.0
- **ngrok URL changed** → Free ngrok URLs change on restart, update `.env` and Twilio webhook

## Architecture

Brief description of the pipeline: Twilio → ngrok → FastAPI → Pipecat → (Whisper STT → Llama LLM → Cartesia TTS) → Twilio → Caller's phone

## Call Logs
After each call, a JSON transcript is automatically saved to the `call_logs/` directory. Each file contains:
- Caller's phone number
- Call start/end time and duration
- Full conversation transcript with timestamps
- AI-generated summary with lead score, sentiment, and follow-up recommendation

## Viewing Call Logs
```bash
# Terminal viewer
python call_viewer.py

# API endpoint (while server is running)
curl http://localhost:8765/calls
```

## API Endpoints
- `POST /incoming-call` — Twilio webhook (do not call directly)
- `GET /ws` — WebSocket for Twilio audio stream (do not call directly)
- `GET /health` — Server health check
- `GET /calls` — List recent call logs with summaries
- `GET /calls/{filename}` — Get specific call details
