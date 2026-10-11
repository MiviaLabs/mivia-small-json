# Contributing

Thanks for looking at mivia-small-json. Read [AGENTS.md](AGENTS.md) first —
it carries the engineering rules this repo is run by. The short version:

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
python scripts/assemble_models.py
```

## Rules that get PRs merged

- Run the focused test files for the surfaces you touch; run the full
  suite (`pytest -q`) once before you commit.
- New dependencies need an exact pin in `pyproject.toml` and a one-line
  justification.
- No pickle — persistence is skops.io only.
- No third-party dataset enters a commit (see [docs/DATA.md](docs/DATA.md));
  builders fetch on demand.
- Model changes need a before/after measurement and a
  [docs/MODEL-STATE.md](docs/MODEL-STATE.md) update in the same change.
- Numbers quote the set they were measured on.

## PRs

Keep PRs single-purpose. CI runs install-from-pins plus the full suite and
lint; green CI is required.
