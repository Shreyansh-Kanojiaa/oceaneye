# OCEAN-EYE

Attribute a satellite-observed oil slick to the vessel most likely to have released it,
as a calibrated probability with an explicit unknown-source hypothesis.

SIH 2026 · SIH26143 · Team Kernel Panic. See `CLAUDE.md`.

Scenarios are **synthetic**. Output is a ranked probability for investigation, not a claim
that a vessel caused a spill.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
