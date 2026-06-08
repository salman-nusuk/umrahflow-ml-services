"""Retry production OCR for any entries currently marked as an error in
production.json. Runs sequentially with a small sleep to avoid overloading
the prod sidecar."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_ocr import run_production_ocr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--production", required=True, type=Path)
    ap.add_argument("--endpoint", default="https://passport.delveon.com/api/ocr")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    data = json.loads(args.production.read_text())
    todo = [k for k, v in data.items() if "error" in (v.get("fields") or {})]
    print(f"retrying {len(todo)} errored entries", file=sys.stderr)
    fixed = 0
    still_failing = 0
    for i, fid in enumerate(todo, 1):
        path = args.src / fid
        if not path.exists():
            continue
        result = run_production_ocr(args.endpoint, path)
        data[fid] = {"fields": result}
        if "error" in result:
            still_failing += 1
            print(f"  [{i}/{len(todo)}] still failing: {fid} -- {result['error'][:80]}",
                  file=sys.stderr)
        else:
            fixed += 1
            print(f"  [{i}/{len(todo)}] fixed: {fid}", file=sys.stderr)
        time.sleep(args.sleep)
        if i % 5 == 0:
            args.production.write_text(json.dumps(data, indent=2))
    args.production.write_text(json.dumps(data, indent=2))
    print(f"fixed {fixed}, still failing {still_failing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
