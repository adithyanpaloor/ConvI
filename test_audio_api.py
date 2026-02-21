"""Live API test — POST Trial 1 convi.wav to /api/v1/analyze/audio"""
import httpx
import json

WAV_PATH = "inputs/Trial 1 convi.wav"
URL      = "http://localhost:8000/api/v1/analyze/audio"

print(f"POSTing {WAV_PATH} to {URL} ...")
with open(WAV_PATH, "rb") as f:
    resp = httpx.post(
        URL,
        files={"audio_file": ("Trial 1 convi.wav", f, "audio/wav")},
        data={"domain": "financial_banking"},
        timeout=300.0,
    )

print(f"\n{'='*60}")
print(f"HTTP Status : {resp.status_code}")
print(f"{'='*60}")

try:
    data = resp.json()
    print(json.dumps(data, indent=2))
except Exception:
    print("Raw response:", resp.text[:3000])
