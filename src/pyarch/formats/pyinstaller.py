from __future__ import annotations

from dataclasses import dataclass


PYINSTALLER_MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"

PYINSTALLER_COOKIE_SIZE = 88


@dataclass(frozen=True, slots=True)
class PyInstallerFormat:
    """Description of the PyInstaller CArchive format."""

    cookie_size: int = PYINSTALLER_COOKIE_SIZE
    magic: bytes = PYINSTALLER_MAGIC


CURRENT_FORMAT = PyInstallerFormat()


def get_pyinstaller_format(
    *,
    cookie_size: int = PYINSTALLER_COOKIE_SIZE,
) -> PyInstallerFormat:
    """
    Return the PyInstaller format description.

    Keeping this behind a function makes it possible to add format
    variants later without changing the extractor API.
    """

    return PyInstallerFormat(
        cookie_size=cookie_size,
    )