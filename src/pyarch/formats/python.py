from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PythonFormat:
    """Description of a supported Python bytecode format."""

    major: int
    minor: int
    magic: bytes

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}"


def _magic_for(version: tuple[int, int]) -> bytes:
    """
    Return the CPython bytecode magic number.

    Magic values are deliberately kept in one place so adding a new
    Python version does not require modifying the rest of PyArch.
    """

    magics = {
        # Python 3.8
        (3, 8): bytes.fromhex("550d0d0a"),

        # Python 3.9
        (3, 9): bytes.fromhex("610d0d0a"),

        # Python 3.10
        (3, 10): bytes.fromhex("6f0d0d0a"),

        # Python 3.11
        (3, 11): bytes.fromhex("a70d0d0a"),

        # Python 3.12
        (3, 12): bytes.fromhex("cb0d0d0a"),

        # Python 3.13
        (3, 13): bytes.fromhex("f20d0d0a"),

        # Python 3.14
        (3, 14): bytes.fromhex("e30d0d0a"),
    }

    try:
        return magics[version]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Python version: {version[0]}.{version[1]}"
        ) from error


SUPPORTED_PYTHON_VERSIONS = (
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
)


PYTHON_FORMATS: dict[tuple[int, int], PythonFormat] = {
    version: PythonFormat(
        major=version[0],
        minor=version[1],
        magic=_magic_for(version),
    )
    for version in SUPPORTED_PYTHON_VERSIONS
}


def get_python_format(
    major: int,
    minor: int,
) -> PythonFormat | None:
    """Return the format handler for a Python version."""

    return PYTHON_FORMATS.get((major, minor))


def is_supported_python_version(
    major: int,
    minor: int,
) -> bool:
    """Check whether a Python version is supported."""

    return (major, minor) in PYTHON_FORMATS