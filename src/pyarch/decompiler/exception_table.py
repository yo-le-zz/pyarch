from __future__ import annotations

"""
Parser for CPython's "zero-cost" exception table (`co_exceptiontable`),
introduced in Python 3.11 and unchanged in format through 3.14.

This is a small, self-contained re-implementation of the same
decoding `dis._parse_exception_table` does, kept here rather than
depending on that private `dis` API, and written so it works
identically whether `co_exceptiontable` came from this interpreter's
own CodeType or from a `RemoteCode` dumped by a different one -- the
byte format itself does not vary by version, only the surrounding
opcodes do.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


def _parse_varint(iterator) -> int:
    first = next(iterator)
    value = first & 0x3F

    while first & 0x40:
        first = next(iterator)
        value = (value << 6) | (first & 0x3F)

    return value


def parse_exception_table(
    raw: bytes,
) -> list[ExceptionTableEntry]:
    """
    Decode `co_exceptiontable` bytes into a list of entries.

    Each entry means: if an exception propagates out of instruction
    offsets [start, end), execution jumps to `target`.
    """

    entries: list[ExceptionTableEntry] = []
    iterator = iter(raw)

    try:
        while True:
            start = _parse_varint(iterator) * 2
            length = _parse_varint(iterator) * 2
            end = start + length
            target = _parse_varint(iterator) * 2
            depth_and_lasti = _parse_varint(iterator)
            depth = depth_and_lasti >> 1
            lasti = bool(depth_and_lasti & 1)

            entries.append(
                ExceptionTableEntry(
                    start=start,
                    end=end,
                    target=target,
                    depth=depth,
                    lasti=lasti,
                )
            )
    except StopIteration:
        return entries


def get_exception_table(code) -> list[ExceptionTableEntry]:
    """
    Return the exception table for `code` (a real `CodeType` or a
    `RemoteCode`), or an empty list for code that has none -- either
    because it raises nothing, or because it predates Python 3.11's
    exception-table mechanism (spec section 20's "3.11+ differences").
    """

    raw = getattr(code, "co_exceptiontable", None)

    if not raw:
        return []

    return parse_exception_table(raw)
