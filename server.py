import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

# Configure logging
try:
    logger.remove(0)
except ValueError:
    pass
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

NGROK_URL = os.getenv("NGROK_URL", "").replace("https://", "").replace("http://", "").rstrip("/")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-voice-agent"}

@app.get("/calls")
async def get_calls():
    """Return recent call logs."""
    logs_dir = Path(__file__).parent / "call_logs"
    if not logs_dir.exists():
        return {"calls": []}
    
    calls = []
    for filepath in sorted(logs_dir.glob("call_*.json"), reverse=True)[:20]:
        try:
            call_data = json.loads(filepath.read_text(encoding="utf-8"))
            calls.append(call_data)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
    
    return {"calls": calls, "total": len(calls)}


@app.get("/calls/{filename}")
async def get_call_detail(filename: str):
    """Return a specific call log."""
    filepath = Path(__file__).parent / "call_logs" / filename
    if not filepath.exists():
        return {"error": "Call not found"}
    return json.loads(filepath.read_text(encoding="utf-8"))


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio hits this endpoint when someone calls your number.
    We return TwiML telling Twilio to stream audio to our WebSocket.
    """
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    logger.info(f"Incoming call from: {caller_number}")

    # Return the TwiML instructions directly
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{NGROK_URL}/ws">
            <Parameter name="caller_number" value="{caller_number}" />
        </Stream>
    </Connect>
</Response>"""

    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted from Twilio")

    # Read messages from Twilio until we get the 'start' event with stream_sid
    stream_sid = None
    call_sid = None
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")
            if event == "connected":
                logger.info("Twilio stream connected")
                continue
            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"].get("callSid")
                logger.info(f"Stream started: stream_sid={stream_sid}, call_sid={call_sid}")
                break
    except Exception as e:
        logger.error(f"Error reading Twilio start event: {e}")
        await websocket.close()
        return

    if not stream_sid:
        logger.error("Never received Twilio stream_sid, closing connection")
        await websocket.close()
        return

    try:
        from bot import run_bot
        await run_bot(websocket, stream_sid, call_sid)
    except asyncio.CancelledError:
        logger.info("Call cancelled (caller hung up)")
    except ConnectionError as e:
        logger.warning(f"Connection lost: {e}")
    except Exception as e:
        logger.error(f"Unexpected bot error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket cleanup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
