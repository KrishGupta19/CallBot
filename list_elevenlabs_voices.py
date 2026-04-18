import os
import asyncio

import aiohttp
from dotenv import load_dotenv


API_URL = "https://api.elevenlabs.io/v1/voices"


async def main() -> None:
    # Load variables from local .env so running the script
    # from a shell picks up ELEVENLABS_API_KEY automatically.
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY") or ""
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in environment (.env).")

    headers = {"xi-api-key": api_key}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(API_URL) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise SystemExit(f"ElevenLabs API error: {resp.status} {text[:300]}")
            data = await resp.json()

    voices = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(voices, list) or not voices:
        raise SystemExit("No voices returned from ElevenLabs.")

    # Print minimal, copy/paste-friendly output.
    for v in voices:
        if not isinstance(v, dict):
            continue
        name = (v.get("name") or "").strip()
        vid = (v.get("voice_id") or "").strip()
        category = (v.get("category") or "").strip()
        labels = v.get("labels") if isinstance(v.get("labels"), dict) else {}
        accent = (labels.get("accent") or "").strip()
        gender = (labels.get("gender") or "").strip()
        age = (labels.get("age") or "").strip()

        bits = [b for b in [category, gender, age, accent] if b]
        meta = f" ({', '.join(bits)})" if bits else ""
        if name and vid:
            print(f"{name}{meta}: {vid}")


if __name__ == "__main__":
    asyncio.run(main())

