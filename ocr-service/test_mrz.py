"""Quick smoke test for fastmrz. Pass an image path as argv[1]."""
import sys, json
from fastmrz import FastMRZ

if len(sys.argv) < 2:
    print("usage: python test_mrz.py <image_path>")
    sys.exit(1)

result = FastMRZ().get_details(sys.argv[1])
print(json.dumps(result, indent=2, default=str))
