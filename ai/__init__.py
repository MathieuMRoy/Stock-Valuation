"""AI module public exports with lazy loading.

This keeps light-weight modules such as `ai.models` and `ai.router` importable
without forcing the full Streamlit + ADK runtime to initialize during tests.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ai_analyst_report",
    "build_ai_chat_signature",
    "chat_with_ai_analyst",
    "create_ai_chat_session",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".adk_analyst", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
