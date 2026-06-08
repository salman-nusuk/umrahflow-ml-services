"""Benchmark the hybrid pipeline across multiple passport samples."""
import sys, os, json, base64, time, urllib.request, glob
from concurrent.futures import ThreadPoolExecutor
from fastmrz import FastMRZ

SAMPLES_GLOB = sys.argv[1] if len(sys.argv) > 1 else "/Users/afaqahmad/Downloads/ses 11111/**/*.jpeg"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

VIZ_PROMPT = (
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

mrz_engine = FastMRZ()


def run_mrz(path):
    t0 = time.time()
    try:
        return mrz_engine.get_details(path), time.time() - t0
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}, time.time() - t0


def run_viz(path):
    t0 = time.time()
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": VIZ_PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        return json.loads(resp.get("response", "{}")), time.time() - t0
    except Exception as e:
        return {"error": str(e)}, time.time() - t0


def process(path):
    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_mrz = ex.submit(run_mrz, path)
        f_viz = ex.submit(run_viz, path)
        mrz, t_mrz = f_mrz.result()
        viz, t_viz = f_viz.result()
    sex = mrz.get("sex")
    if sex == "M":
        viz["husband_name"] = None
    elif sex == "F":
        viz["father_name"] = None
    return {
        "file": os.path.basename(path),
        "mrz_status": mrz.get("status"),
        "t_mrz": round(t_mrz, 2),
        "t_viz": round(t_viz, 2),
        "t_wall": round(time.time() - t, 2),
        "mrz": {
            "name": f"{mrz.get('surname','')} {mrz.get('given_name','')}".strip(),
            "passport": mrz.get("document_number"),
            "dob": mrz.get("birth_date"),
            "sex": sex,
            "expiry": mrz.get("expiry_date"),
            "cnic_mrz": mrz.get("optional_data"),
        },
        "viz": viz,
    }


paths = sorted(glob.glob(SAMPLES_GLOB, recursive=True))[:N]
print(f"# benchmarking {len(paths)} samples\n")

results = []
total = time.time()
for i, p in enumerate(paths, 1):
    print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
    r = process(p)
    results.append(r)
    print(f"   mrz={r['mrz_status']} t_mrz={r['t_mrz']}s t_viz={r['t_viz']}s wall={r['t_wall']}s", flush=True)

print("\n# summary")
ok = sum(1 for r in results if r["mrz_status"] == "SUCCESS")
avg_wall = sum(r["t_wall"] for r in results) / len(results)
avg_viz = sum(r["t_viz"] for r in results) / len(results)
avg_mrz = sum(r["t_mrz"] for r in results) / len(results)
print(f"mrz_success: {ok}/{len(results)}")
print(f"avg t_mrz:  {avg_mrz:.1f}s")
print(f"avg t_viz:  {avg_viz:.1f}s")
print(f"avg t_wall: {avg_wall:.1f}s")
print(f"total:      {time.time()-total:.1f}s")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nfull results -> benchmark_results.json")
