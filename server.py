import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import RedirectResponse
from fastapi.responses import PlainTextResponse, HTMLResponse

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

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")

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
            if isinstance(call_data, list):
                # Legacy format: transcript-only list (sometimes empty). Inject call_start
                # from file modified time so the dashboard can show date/time.
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                calls.append(
                    {
                        "filename": filepath.name,
                        "call_start": mtime,
                        "call_end": None,
                        "duration_seconds": None,
                        "caller_number": None,
                        "business_type": "unknown",
                        "transcript": call_data,
                        "transcript_text": " ".join(
                            [
                                (t.get("text") or "").strip()
                                for t in call_data
                                if isinstance(t, dict)
                            ]
                        ).strip(),
                        "intake": None,
                        "summary": {
                            "caller_name": None,
                            "interest": None,
                            "budget_range": None,
                            "timeline": None,
                            "source": None,
                            "lead_score": None,
                            "sentiment": None,
                            "summary": None,
                            "follow_up_action": None,
                        },
                    }
                )
            else:
                call_data["filename"] = filepath.name
                call_data.setdefault("business_type", "unknown")
                call_data.setdefault("intake", None)
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


@app.get("/dashboard")
async def dashboard():
    """Serve the web dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1><p>Please create dashboard.html</p>", status_code=404)
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))

@app.get("/dashboard/")
async def dashboard_slash():
    return RedirectResponse(url="/dashboard")


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio hits this endpoint when someone calls your number.
    We return TwiML with a simple IVR menu:
    - Press 1 for doctor appointments
    - Press 2 for bakery orders
    """
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    logger.info(f"Incoming call from: {caller_number}")

    base_url = f"https://{NGROK_URL}" if NGROK_URL else ""
    route_url = f"{base_url}/route-call" if base_url else "/route-call"

    # Return the IVR menu
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="dtmf" numDigits="1" timeout="7" action="{route_url}" method="POST">
        <Say voice="alice">Welcome. For doctor appointments, press 1. For bakery orders, press 2.</Say>
    </Gather>
    <Say voice="alice">Sorry, I did not receive your selection.</Say>
    <Redirect method="POST">/incoming-call</Redirect>
</Response>"""

    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.post("/route-call")
async def route_call(request: Request):
    """Handle IVR digit and start streaming to /ws with business_type."""
    form_data = await request.form()
    digits = (form_data.get("Digits") or "").strip()
    caller_number = form_data.get("From", "unknown")

    business_type = "doctor" if digits == "1" else "bakery" if digits == "2" else "unknown"
    logger.info(f"IVR selection digits={digits!r} -> business_type={business_type} caller={caller_number}")

    if business_type == "unknown":
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Invalid selection.</Say>
    <Redirect method="POST">/incoming-call</Redirect>
</Response>"""
        return PlainTextResponse(content=twiml, media_type="application/xml")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{NGROK_URL}/ws">
            <Parameter name="caller_number" value="{caller_number}" />
            <Parameter name="business_type" value="{business_type}" />
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
    caller_number = None
    business_type = None
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
                custom = data["start"].get("customParameters") or {}
                caller_number = custom.get("caller_number") or custom.get("From") or None
                business_type = custom.get("business_type") or None
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
        await run_bot(websocket, stream_sid, call_sid, caller_number, business_type)
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
