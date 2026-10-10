# Test fixtures

- `edinet_doclist_sample.json` — **synthetic** EDINET document list (fake companies `サンプル…`,
  fake EDINET codes `E99xxx`) with the real field names of API v2. Covers: an annual report,
  a third-party tender offer, an issuer self-tender (must be excluded) and a withdrawn filing.

Rules for fixtures:
- Keep them small (a few KB) and synthetic. Never commit full real filings or price files.
- When a real filing reveals a new format quirk, reproduce the quirk in a tiny synthetic fixture
  and add a test for it — that is how the parser gets more robust over time.
