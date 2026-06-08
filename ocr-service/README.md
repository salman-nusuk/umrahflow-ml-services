# OCR Service

Isolated Python environment for passport OCR. MRZ extraction via [fastmrz](https://github.com/sivakumar-mahalingam/fastmrz) (Tesseract + custom ONNX segmentation).

## Setup (already done on this machine)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

System deps (already present):
- `tesseract` 5.5.1 (`brew install tesseract`)
- `mrz.traineddata` installed in `/opt/homebrew/share/tessdata/`

## Smoke test

```bash
source venv/bin/activate
python test_mrz.py /path/to/passport.jpg
```

Returns JSON with surname, given names, document number, nationality, DOB, sex, expiry, and check-digit validity.
