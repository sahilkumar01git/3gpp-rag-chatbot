"""
Backward-compatible entry point — the original project was run with
`python main.py`. All logic now lives in `app/`; this just delegates to
the CLI so existing muscle memory / scripts keep working.

Prefer `python -m app.cli` directly, or `uvicorn app.api.main:app --reload`
for the full FastAPI + web UI experience (see README.md).
"""

from app.cli import main

if __name__ == "__main__":
    main()
