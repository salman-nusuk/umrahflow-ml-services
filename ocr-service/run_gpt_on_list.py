"""Run fastmrz + GPT-5.4 Mini pipeline on a file list, write merged JSON."""
import sys, os, json
from run_pipeline_openai import process

FILE_LIST = sys.argv[1]
OUT = sys.argv[2]

paths = [l.strip() for l in open(FILE_LIST) if l.strip()]
results = []
for i, p in enumerate(paths, 1):
    print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
    r = process(p)
    # flatten viz fields up
    flat = {
        "file": r["file"],
        "mrz_status": r["mrz_status"],
        "surname": r["mrz"].get("name", "").split(" ", 1)[0] or None,
        "given_names": " ".join(r["mrz"].get("name", "").split(" ")[1:]) or None,
        "passport_number": r["mrz"].get("passport"),
        "date_of_birth": r["mrz"].get("dob"),
        "sex": r["mrz"].get("sex"),
        "expiry_date": r["mrz"].get("expiry"),
        "cnic_mrz": r["mrz"].get("cnic_mrz"),
    }
    flat.update(r.get("viz") or {})
    flat["t_wall"] = r["t_wall"]
    results.append(flat)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}")
