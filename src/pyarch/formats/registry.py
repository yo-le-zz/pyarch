from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FormatInfo:
    """Generic binary format description."""

    name: str
    handler: Any


_REGISTRY: dict[str, FormatInfo] = {}


def register_format(
    name: str,
    handler: Any,
) -> None:
    """Register a format handler."""

    if name in _REGISTRY:
        raise ValueError(f"Format already registered: {name}")

    _REGISTRY[name] = FormatInfo(
        name=name,
        handler=handler,
    )


def get_format(name: str) -> FormatInfo | None:
    """Get a registered format."""

    return _REGISTRY.get(name)


def registered_formats() -> tuple[FormatInfo, ...]:
    """Return all registered formats."""

    return tuple(_REGISTRY.values())