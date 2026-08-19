"""MAGPIE-native transport for the LuxAI speech-to-speech runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import S2SAudioFrame

__version__ = "0.1.0"

__all__ = ["S2SAudioFrame", "__version__"]


def __getattr__(name: str) -> Any:
    """Avoid importing MAGPIE transport dependencies for utility commands."""

    if name == "S2SAudioFrame":
        from .protocol import S2SAudioFrame

        return S2SAudioFrame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
