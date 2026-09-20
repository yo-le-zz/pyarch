from __future__ import annotations

"""
Python bytecode version detection (PyArch spec section 8).

Maps a .pyc's magic number to the CPython version that produced it,
so PyArch never assumes the analyzed bytecode matches whatever
interpreter happens to be running PyArch itself.

Magic numbers are the 2-byte value CPython embeds at the very start
of every .pyc; the surrounding 2 bytes (``\\r\\n``) are a load-time
sanity check, not part of the version number. Values below are the
canonical numbers for each *released* minor version, taken from
``Lib/importlib/_bootstrap_external.py`` in the CPython source for
3.8-3.13, and ``Include/internal/pycore_magic_number.h`` for 3.14.

A few Python versions bumped the magic number more than once during
their own beta/rc cycle; this table only lists the number for the
final, released version of each line, which is what PyArch is meant
to target (spec section 3).
"""

MAGIC_TO_VERSION: dict[int, tuple[int, int]] = {
    3413: (3, 8),
    3425: (3, 9),
    3439: (3, 10),
    3495: (3, 11),
    3531: (3, 12),
    3571: (3, 13),
    3627: (3, 14),
}


class VersionDetectionError(Exception):
    """Raised when a .pyc's magic number is not recognized."""


def detect_version(magic: bytes) -> tuple[int, int]:
    """
    Return the ``(major, minor)`` Python version for a .pyc's magic
    bytes (the first 2 bytes of the 4-byte magic number field).

    Raises `VersionDetectionError` for an unrecognized magic number
    rather than guessing -- an unsupported/future/corrupt magic
    number should be reported, not silently treated as some other
    version.
    """

    if len(magic) < 2:
        raise VersionDetectionError(
            "Magic number is too short to identify a Python version."
        )

    raw = int.from_bytes(magic[:2], "little")

    version = MAGIC_TO_VERSION.get(raw)

    if version is None:
        raise VersionDetectionError(
            f"Unrecognized .pyc magic number ({raw}). This may be a "
            f"Python version PyArch does not yet know about, or a "
            f"corrupted file."
        )

    return version
