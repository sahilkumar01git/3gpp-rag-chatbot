"""
Optional LangSmith tracing.

Off by default. To enable, set in `.env` (or the environment):

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=3gpp-rag-chatbot   # optional, this is the default

`traceable` is LangSmith's decorator when the `langsmith` package is
installed, and a no-op otherwise — so the pipeline runs unchanged without
the package or the key. LangSmith itself also records nothing unless
LANGSMITH_TRACING is true.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# The LangSmith SDK reads os.environ directly, whereas the rest of the app
# reads `.env` through pydantic-settings — load it here so both agree.
load_dotenv()
os.environ.setdefault("LANGSMITH_PROJECT", "3gpp-rag-chatbot")

try:
    from langsmith import traceable
except ImportError:  # langsmith not installed — tracing silently disabled

    def traceable(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))
