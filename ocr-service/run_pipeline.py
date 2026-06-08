"""Hybrid passport OCR: fastmrz (MRZ) + Qwen2.5-VL (VIZ), parallel, with timing."""
import sys, json, base64, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

if len(sys.argv) < 2:
    print("usage: python run_pipeline.py <image>")
    sys.exit(1)

img_path = sys.argv[1]


def run_mrz():
    t0 = time.time()
    from fastmrz import FastMRZ
    result = FastMRZ().get_details(img_path)
    return result, time.time() - t0


def run_viz():
    t0 = time.time()
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    prompt = (
        "Extract fields from this Pakistani passport image. Return ONLY valid JSON.\n"
        "Rules:\n"
        "- Use null for any field not clearly visible. Do NOT guess.\n"
        "- place_of_birth and place_of_issue must be CITIES (e.g. LAHORE, KARACHI, ISLAMABAD), never the country name.\n"
        "- father_name is the value next to 'Father Name' label only; do not copy the holder's name.\n"
        "- husband_name applies only if the holder is female; otherwise null.\n"
        "- Dates as YYYY-MM-DD.\n"
        "- booklet_number is the small serial usually printed top-right (e.g. M9161105).\n"
        "- cnic is a 13-digit number sometimes formatted XXXXX-XXXXXXX-X.\n"
        "Keys: place_of_birth, place_of_issue, date_of_issue, father_name, husband_name, booklet_number, cnic."
    )
    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return json.loads(resp.get("response", "{}")), time.time() - t0


t_total = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    f_mrz = ex.submit(run_mrz)
    f_viz = ex.submit(run_viz)
    mrz, t_mrz = f_mrz.result()
    viz, t_viz = f_viz.result()
elapsed = time.time() - t_total

sex = mrz.get("sex")
if sex == "M":
    viz["husband_name"] = None
elif sex == "F":
    viz["father_name"] = None

merged = {
    # MRZ-derived (check-digit validated)
    "surname": mrz.get("surname"),
    "given_names": mrz.get("given_name"),
    "passport_number": mrz.get("document_number"),
    "nationality": mrz.get("nationality_code"),
    "date_of_birth": mrz.get("birth_date"),
    "sex": mrz.get("sex"),
    "expiry_date": mrz.get("expiry_date"),
    "cnic_mrz": mrz.get("optional_data"),
    # VIZ-derived (Qwen)
    "place_of_birth": viz.get("place_of_birth"),
    "place_of_issue": viz.get("place_of_issue"),
    "date_of_issue": viz.get("date_of_issue"),
    "father_name": viz.get("father_name"),
    "husband_name": viz.get("husband_name"),
    "booklet_number": viz.get("booklet_number"),
    "cnic": viz.get("cnic"),
}

print(json.dumps({
    "timing": {
        "mrz_seconds": round(t_mrz, 2),
        "viz_seconds": round(t_viz, 2),
        "wall_seconds": round(elapsed, 2),
    },
    "mrz_status": mrz.get("status"),
    "fields": merged,
}, indent=2, ensure_ascii=False))
