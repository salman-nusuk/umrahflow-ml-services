# Fixtures

Drop your real WhatsApp fixtures under this directory (or symlink them).
The runner serves whatever lives here over a local `http.server` on port 8765
so the agent can OCR them like real Twilio-hosted media.

Expected layout (matches scenarios.yml media: paths):

```
fixtures/
├── manifest.yml
├── AGOG/
│   ├── Hotel Voucher.pdf
│   ├── passport_1.jpg
│   ├── passport_2.jpg
│   ├── passport_3.jpg
│   ├── passport_4.jpg
│   └── passport_5.jpg
├── BAAB E KABA/
│   ├── Hotel Voucher.pdf
│   └── passport_1..5.jpg
├── corrupt/
│   ├── blank.jpg
│   └── blank2.jpg
├── stray/
│   └── unknown_passport.jpg
├── shared/
│   └── shared_passport.jpg
├── cross-agent/
│   ├── AB1234567.jpg
│   └── UB-T999002-voucher.pdf
├── blacklist/
│   └── AB1234567.jpg
└── edge/
    ├── voucher-no-mutamers.pdf
    ├── passport-empty-pno.jpg
    ├── voucher-AB1234567.pdf
    └── passport-AB1234567-padded.jpg
```

- `manifest.yml` records each fixture's expected OCR output.
- `python -m tests.agent.runner --probe-fixtures` prints the OCR for each
  file so you can paste the real `ub_number` + `expected_passport_numbers`
  into manifest.yml.
- DO NOT commit the binary fixtures. The repo `.gitignore` covers
  `tests/agent/fixtures/*` (everything except this README and manifest.yml).
