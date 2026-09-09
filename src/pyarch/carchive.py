from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


class CArchiveError(Exception):
    """Raised when a PyInstaller CArchive cannot be parsed."""


@dataclass(slots=True)
class CArchiveEntry:
    """A single entry from a PyInstaller CArchive TOC."""

    position: int
    compressed_size: int
    uncompressed_size: int
    compressed: bool
    typecode: str
    name: str


@dataclass(slots=True)
class CArchive:
    """Parsed PyInstaller CArchive."""

    path: Path
    package_offset: int
    package_length: int
    toc_offset: int
    toc_length: int
    python_version: int
    python_library: str
    entries: list[CArchiveEntry]


# structlen, dpos, dlen, ulen, flag, typecode
TOC_HEADER = struct.Struct("!iiiibc")


EXTRACTABLE_TYPES = {
    "b",  # binary
    "x",  # data
    "z",  # zip
    "s",  # symlink
    "d",  # dependency
}


def parse_toc(
    file,
    *,
    toc_offset: int,
    toc_length: int,
) -> list[CArchiveEntry]:
    """Parse the CArchive table of contents."""

    file.seek(toc_offset)
    toc = file.read(toc_length)

    if len(toc) != toc_length:
        raise CArchiveError("Could not read the complete CArchive TOC.")

    entries: list[CArchiveEntry] = []
    offset = 0

    while offset < len(toc):
        if len(toc) - offset < TOC_HEADER.size:
            raise CArchiveError("Truncated CArchive TOC entry.")

        (
            struct_length,
            data_position,
            compressed_size,
            uncompressed_size,
            compression_flag,
            typecode,
        ) = TOC_HEADER.unpack_from(toc, offset)

        if struct_length < TOC_HEADER.size:
            raise CArchiveError(
                f"Invalid CArchive TOC entry size: {struct_length}"
            )

        end = offset + struct_length

        if end > len(toc):
            raise CArchiveError("CArchive TOC entry exceeds TOC size.")

        name_bytes = toc[
            offset + TOC_HEADER.size : end
        ]

        name = name_bytes.rstrip(b"\0").decode(
            "utf-8",
            errors="replace",
        )

        entries.append(
            CArchiveEntry(
                position=data_position,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                compressed=compression_flag != 0,
                typecode=typecode.decode(
                    "ascii",
                    errors="replace",
                ),
                name=name,
            )
        )

        offset = end

    return entries


def read_entry(
    archive: CArchive,
    entry: CArchiveEntry,
) -> bytes:
    """Read and decompress a CArchive entry."""

    with archive.path.open("rb") as file:
        file.seek(archive.package_offset + entry.position)

        data = file.read(entry.compressed_size)

    if len(data) != entry.compressed_size:
        raise CArchiveError(
            f"Could not read CArchive entry: {entry.name}"
        )

    if entry.compressed:
        try:
            data = zlib.decompress(data)
        except zlib.error as error:
            raise CArchiveError(
                f"Could not decompress CArchive entry: {entry.name}"
            ) from error

    if len(data) != entry.uncompressed_size:
        raise CArchiveError(
            f"Invalid size for CArchive entry: {entry.name}"
        )

    return data


def parse_carchive(
    path: Path,
    *,
    package_offset: int,
    package_length: int,
    toc_offset: int,
    toc_length: int,
    python_version: int,
    python_library: str,
) -> CArchive:
    """Parse a CArchive using already decoded cookie information."""

    with path.open("rb") as file:
        entries = parse_toc(
            file,
            toc_offset=package_offset + toc_offset,
            toc_length=toc_length,
        )

    return CArchive(
        path=path,
        package_offset=package_offset,
        package_length=package_length,
        toc_offset=toc_offset,
        toc_length=toc_length,
        python_version=python_version,
        python_library=python_library,
        entries=entries,
    )


def extract_entry(
    archive: CArchive,
    entry: CArchiveEntry,
    output: Path,
) -> Path:
    """Extract one CArchive entry to disk."""

    if entry.typecode not in EXTRACTABLE_TYPES:
        raise CArchiveError(
            f"Entry type {entry.typecode!r} is not directly extractable: "
            f"{entry.name}"
        )

    target = output / entry.name
    target.parent.mkdir(parents=True, exist_ok=True)

    data = read_entry(archive, entry)

    target.write_bytes(data)

    return target


def extract_all(
    archive: CArchive,
    output: Path,
) -> list[Path]:
    """Extract all extractable CArchive entries."""

    output.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []

    for entry in archive.entries:
        if entry.typecode not in EXTRACTABLE_TYPES:
            continue

        extracted.append(
            extract_entry(
                archive,
                entry,
                output,
            )
        )

    return extracted