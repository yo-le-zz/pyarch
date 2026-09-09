from __future__ import annotations

import marshal
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


class PYZError(Exception):
    """Raised when a PYZ archive cannot be parsed."""


PYZ_MAGIC = b"PYZ\x00"
PYZ_HEADER = struct.Struct("!4s4si")


@dataclass(slots=True)
class PYZEntry:
    """A single entry from a PyInstaller PYZ archive."""

    name: str
    typecode: int
    position: int
    length: int


@dataclass(slots=True)
class PYZArchive:
    """Parsed PyInstaller PYZ archive."""

    path: Path
    python_magic: bytes
    toc_offset: int
    entries: list[PYZEntry]


def parse_pyz(path: Path) -> PYZArchive:
    """Parse a PyInstaller PYZ archive."""

    path = path.resolve()

    if not path.is_file():
        raise PYZError(
            f"PYZ archive not found: {path}"
        )

    file_size = path.stat().st_size

    if file_size < PYZ_HEADER.size:
        raise PYZError(
            "PYZ archive is too small."
        )

    with path.open("rb") as file:
        header = file.read(PYZ_HEADER.size)

        if len(header) != PYZ_HEADER.size:
            raise PYZError(
                "Incomplete PYZ header."
            )

        magic, python_magic, toc_offset = (
            PYZ_HEADER.unpack(header)
        )

        if magic != PYZ_MAGIC:
            raise PYZError(
                f"Invalid PYZ magic: {magic!r}"
            )

        if toc_offset < PYZ_HEADER.size:
            raise PYZError(
                f"Invalid PYZ TOC offset: {toc_offset}"
            )

        if toc_offset >= file_size:
            raise PYZError(
                "PYZ TOC lies outside the archive."
            )

        file.seek(toc_offset)
        toc_data = file.read()

    try:
        toc = marshal.loads(toc_data)
    except (
        EOFError,
        ValueError,
        TypeError,
    ) as error:
        raise PYZError(
            "Could not decode PYZ TOC."
        ) from error

    # PyInstaller stores the TOC as a list of:
    #
    #     (module_name, (typecode, position, length))
    #
    # Older implementations sometimes expose this as
    # a dictionary, so support both representations.
    if isinstance(toc, dict):
        raw_entries = list(toc.items())

    elif isinstance(toc, list):
        raw_entries = toc

    else:
        raise PYZError(
            f"Invalid PYZ TOC type: {type(toc).__name__}"
        )

    entries: list[PYZEntry] = []

    for item in raw_entries:
        if not isinstance(item, (tuple, list)):
            raise PYZError(
                "Invalid PYZ TOC entry."
            )

        if len(item) != 2:
            raise PYZError(
                "Invalid PYZ TOC entry length."
            )

        name, value = item

        if not isinstance(name, str):
            raise PYZError(
                "PYZ TOC contains a non-string module name."
            )

        if not isinstance(value, (tuple, list)):
            raise PYZError(
                f"Invalid PYZ entry: {name!r}"
            )

        if len(value) != 3:
            raise PYZError(
                f"Invalid PYZ entry: {name!r}"
            )

        typecode, position, length = value

        if not isinstance(typecode, int):
            raise PYZError(
                f"Invalid PYZ type: {name!r}"
            )

        if not isinstance(position, int):
            raise PYZError(
                f"Invalid PYZ position: {name!r}"
            )

        if not isinstance(length, int):
            raise PYZError(
                f"Invalid PYZ length: {name!r}"
            )

        if position < PYZ_HEADER.size:
            raise PYZError(
                f"Invalid PYZ position: {name!r}"
            )

        if length <= 0:
            raise PYZError(
                f"Invalid PYZ length: {name!r}"
            )

        if position + length > toc_offset:
            raise PYZError(
                f"PYZ entry exceeds data section: {name!r}"
            )

        entries.append(
            PYZEntry(
                name=name,
                typecode=typecode,
                position=position,
                length=length,
            )
        )

    return PYZArchive(
        path=path,
        python_magic=python_magic,
        toc_offset=toc_offset,
        entries=entries,
    )


def read_entry(
    archive: PYZArchive,
    entry: PYZEntry,
) -> bytes:
    """Read and decompress a PYZ entry."""

    with archive.path.open("rb") as file:
        file.seek(entry.position)
        compressed = file.read(entry.length)

    if len(compressed) != entry.length:
        raise PYZError(
            f"Could not read PYZ entry: {entry.name}"
        )

    try:
        return zlib.decompress(compressed)
    except zlib.error as error:
        raise PYZError(
            f"Could not decompress PYZ entry: {entry.name}"
        ) from error


def _module_path(name: str) -> Path:
    """Convert a Python module name to a safe filesystem path."""

    normalized = name.replace("\\", "/")

    parts = [
        part
        for part in normalized.split("/")
        if part not in {"", ".", ".."}
    ]

    if not parts:
        raise PYZError(
            f"Invalid PYZ module name: {name!r}"
        )

    return Path(*parts)


def _make_pyc(
    python_magic: bytes,
    code: bytes,
) -> bytes:
    """Wrap a marshalled code object in a Python 3.7+ pyc header."""

    if len(python_magic) != 4:
        raise PYZError(
            "Invalid Python magic in PYZ header."
        )

    return (
        python_magic
        + b"\x00\x00\x00\x00"
        + b"\x00" * 8
        + code
    )


def extract_pyz(
    archive: PYZArchive,
    output: Path,
) -> list[Path]:
    """Extract PYZ entries as .pyc files."""

    output = output.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted: list[Path] = []

    for entry in archive.entries:
        data = read_entry(
            archive,
            entry,
        )

        relative = _module_path(
            entry.name
        )

        # Package/module naming:
        #
        # package.submodule -> package/submodule.pyc
        #
        # The PYZ TOC contains module names, not paths.
        parts = entry.name.split(".")

        if len(parts) > 1:
            relative = Path(
                *parts
            ).with_suffix(".pyc")
        else:
            relative = relative.with_suffix(".pyc")

        target = (
            output / relative
        ).resolve()

        try:
            target.relative_to(output)
        except ValueError as error:
            raise PYZError(
                f"Unsafe PYZ path: {entry.name!r}"
            ) from error

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_bytes(
            _make_pyc(
                archive.python_magic,
                data,
            )
        )

        extracted.append(target)

    return extracted
