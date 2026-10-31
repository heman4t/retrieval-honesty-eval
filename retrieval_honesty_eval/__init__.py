"""Offline, deterministic evaluation of retrieval honesty for tool-using agents.

The scoring core (``corpus``, ``cases``, ``scoring``) is stdlib-only. The
Verifiers taskset is imported only when asked for, which is how ``prime eval``
resolves it through ``__all__``.
"""

__all__ = ["RetrievalHonestyTaskset"]


def __getattr__(name: str):
    if name == "RetrievalHonestyTaskset":
        from .environment import RetrievalHonestyTaskset

        return RetrievalHonestyTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
