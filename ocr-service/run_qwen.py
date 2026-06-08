"""Run Qwen2.5-VL via Ollama on a passport image, ask for structured JSON."""
import sys, json, base64, time, urllib.request

if len(sys.argv) < 2:
    print("usage: python run_qwen.py <image>")
    sys.exit(1)

with open(sys.argv[1], "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

prompt = (
    "Extract passport fields from this image. Return ONLY valid JSON with these keys "
    "(use null if not visible): surname, given_names, passport_number, nationality, "
    "date_of_birth, sex, expiry_date, place_of_birth, place_of_issue, date_of_issue, "
    "father_name, husband_name, booklet_number, cnic. Dates as YYYY-MM-DD."
)

payload = {
    "model": "qwen2.5vl:7b",
    "prompt": prompt,
    "images": [img_b64],
    "stream": False,
    "format": "json",
    "options": {"temperature": 0},
}

t0 = time.time()
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.loads(r.read())
elapsed = time.time() - t0

print(f"# elapsed: {elapsed:.1f}s")
print(resp.get("response", ""))
