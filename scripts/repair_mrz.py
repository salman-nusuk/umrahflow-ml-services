"""Re-extract just the MRZ part of production.json using the corrected fastmrz
field mapping. Keeps the existing VIZ entries unchanged (so we don't burn
another OpenAI run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_ocr import run_production_mrz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--production", required=True, type=Path)
    args = ap.parse_args()

    data = json.loads(args.production.read_text())
    repaired = 0
    for fid, entry in data.items():
        path = args.src / fid
        if not path.exists():
            continue
        mrz = run_production_mrz(path)
        entry.setdefault("fields", {})["mrz"] = mrz
        repaired += 1
        if repaired % 50 == 0:
            print(f"  {repaired}/{len(data)}", file=sys.stderr)
    args.production.write_text(json.dumps(data, indent=2))
    print(f"repaired {repaired} entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
