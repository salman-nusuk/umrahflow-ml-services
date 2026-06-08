"""Compare GPT pipeline output to agent-produced ground truth."""
import json, glob, os, re, sys

GT_DIR = "/Users/afaqahmad/Documents/umrahflow-dashboard/umrahflow/ocr-service/ground_truth"
GPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "gpt_results.json"

FIELDS = [
    "surname", "given_names", "passport_number", "date_of_birth", "sex",
    "expiry_date", "place_of_birth", "place_of_issue", "date_of_issue",
    "father_name", "husband_name", "booklet_number", "cnic", "tracking_number",
]


def norm(v):
    if v is None or v == "":
        return None
    s = str(v).strip().upper()
    s = re.sub(r"[,\s]+(PAK|PAKISTAN)$", "", s)  # drop trailing country tag
    s = re.sub(r"[\s,.-]+", "", s)  # strip separators for fair compare
    return s or None


gt = []
for f in sorted(glob.glob(f"{GT_DIR}/batch_*.json")):
    gt.extend(json.load(open(f)))
gt_by = {r["file"]: r for r in gt}

gpt = json.load(open(GPT_PATH))
gpt_by = {r["file"]: r for r in gpt}

per_field = {f: {"match": 0, "miss": 0, "wrong": 0, "extra": 0, "both_null": 0} for f in FIELDS}
rows = []

for fname in sorted(set(gt_by) | set(gpt_by)):
    g = gt_by.get(fname, {})
    p = gpt_by.get(fname, {})
    if g.get("document_type") and g["document_type"] != "passport":
        rows.append((fname, g["document_type"], "non-passport: skipped"))
        continue
    line = []
    for f in FIELDS:
        gv, pv = norm(g.get(f)), norm(p.get(f))
        if gv is None and pv is None:
            per_field[f]["both_null"] += 1
            line.append(f"{f}=∅")
        elif gv == pv:
            per_field[f]["match"] += 1
            line.append(f"{f}✓")
        elif gv is None and pv is not None:
            per_field[f]["extra"] += 1
            line.append(f"{f}!extra({p.get(f)!r})")
        elif gv is not None and pv is None:
            per_field[f]["miss"] += 1
            line.append(f"{f}!miss(gt={g.get(f)!r})")
        else:
            per_field[f]["wrong"] += 1
            line.append(f"{f}!wrong(gt={g.get(f)!r} gpt={p.get(f)!r})")
    rows.append((fname, "passport", " ".join(line)))

print("=" * 100)
print(f"{'FIELD':20} {'MATCH':>6} {'MISS':>6} {'WRONG':>6} {'EXTRA':>6} {'∅/∅':>6} {'ACC%':>6}")
print("-" * 100)
total_match = total_total = 0
for f in FIELDS:
    s = per_field[f]
    seen = s["match"] + s["miss"] + s["wrong"]  # exclude both-null and extra from accuracy denominator
    acc = (s["match"] / seen * 100) if seen else 0
    total_match += s["match"]
    total_total += seen
    print(f"{f:20} {s['match']:>6} {s['miss']:>6} {s['wrong']:>6} {s['extra']:>6} {s['both_null']:>6} {acc:>5.0f}%")
print("-" * 100)
overall = (total_match / total_total * 100) if total_total else 0
print(f"{'OVERALL (when GT has value)':20}                                              {overall:>5.0f}%")
print("=" * 100)
print()
print("# per-file detail")
for fname, dtype, line in rows:
    print(f"\n{fname[:80]}  [{dtype}]")
    print(f"  {line}")
