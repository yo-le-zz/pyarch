from __future__ import annotations

import json
import platform
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from .carchive import (
    CArchive,
    CArchiveError,
    extract_entry,
    parse_carchive,
)
from .formats import get_pyinstaller_format
from .pyz import (
    PYZError,
    extract_pyz,
    parse_pyz,
)


class ExtractionError(Exception):
    """Raised when a PyInstaller bundle cannot be extracted."""


COOKIE_FORMAT = "!8siiii64s"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArchiveInfo:
    """Information stored in the PyInstaller CArchive cookie."""

    cookie_offset: int
    package_offset: int
    package_length: int
    toc_offset: int
    toc_length: int
    python_version: int
    python_library: str


@dataclass(slots=True)
class BundleInfo:
    """Detected PyInstaller bundle."""

    executable: Path
    root: Path
    onefile: bool
    archive: ArchiveInfo
    carchive: CArchive | None = None


# ---------------------------------------------------------------------------
# Cookie
# ---------------------------------------------------------------------------


def _find_cookie(path: Path) -> int:
    """
    Find the PyInstaller CArchive cookie.

    PyInstaller stores the cookie near the end of the executable.
    We search backwards instead of assuming it is exactly at EOF so
    that binaries with additional trailing data can still be handled.
    """

    format_info = get_pyinstaller_format()

    magic = format_info.magic
    file_size = path.stat().st_size

    # A PyInstaller executable should have the cookie near the end.
    # Searching the last 1 MiB is cheap and handles normal bundles.
    search_size = min(
        file_size,
        1024 * 1024,
    )

    with path.open("rb") as file:
        file.seek(
            file_size - search_size
        )

        data = file.read(search_size)

    offset = data.rfind(magic)

    if offset == -1:
        raise ExtractionError(
            "PyInstaller archive cookie was not found."
        )

    return (
        file_size
        - search_size
        + offset
    )


def read_archive_info(
    path: Path,
) -> ArchiveInfo:
    """Read and validate the PyInstaller CArchive cookie."""

    cookie_offset = _find_cookie(path)

    format_info = get_pyinstaller_format()
    cookie_size = format_info.cookie_size

    with path.open("rb") as file:
        file.seek(cookie_offset)

        raw_cookie = file.read(
            cookie_size
        )

    if len(raw_cookie) != cookie_size:
        raise ExtractionError(
            "Incomplete PyInstaller archive cookie."
        )

    (
        magic,
        package_length,
        toc_offset,
        toc_length,
        python_version,
        python_library,
    ) = struct.unpack(
        COOKIE_FORMAT,
        raw_cookie,
    )

    if magic != format_info.magic:
        raise ExtractionError(
            "Invalid PyInstaller archive cookie."
        )

    if package_length <= 0:
        raise ExtractionError(
            "Invalid PyInstaller package length."
        )

    if toc_offset < 0:
        raise ExtractionError(
            "Invalid PyInstaller TOC offset."
        )

    if toc_length < 0:
        raise ExtractionError(
            "Invalid PyInstaller TOC length."
        )

    if toc_offset + toc_length > package_length:
        raise ExtractionError(
            "PyInstaller TOC lies outside the package."
        )

    package_offset = (
        cookie_offset
        + cookie_size
        - package_length
    )

    if package_offset < 0:
        raise ExtractionError(
            "Invalid PyInstaller package offset."
        )

    file_size = path.stat().st_size

    if package_offset + package_length > file_size:
        raise ExtractionError(
            "PyInstaller package exceeds executable size."
        )

    python_library_name = (
        python_library
        .rstrip(b"\0")
        .decode(
            "utf-8",
            errors="replace",
        )
    )

    return ArchiveInfo(
        cookie_offset=cookie_offset,
        package_offset=package_offset,
        package_length=package_length,
        toc_offset=toc_offset,
        toc_length=toc_length,
        python_version=python_version,
        python_library=python_library_name,
    )


# ---------------------------------------------------------------------------
# Bundle detection
# ---------------------------------------------------------------------------


def _looks_like_onedir(
    executable: Path,
) -> bool:
    """
    Detect the common PyInstaller onedir layout.

    Modern PyInstaller normally uses:

        application/
        ├── application
        └── _internal/
            ├── python runtime
            ├── libraries
            └── data

    Older versions can place the files directly beside the executable.
    """

    root = executable.parent

    # Modern PyInstaller.
    if (root / "_internal").is_dir():
        return True

    # Common Python runtime files.
    runtime_names = {
        "libpython3.14.so.1.0",
        "libpython3.13.so.1.0",
        "libpython3.12.so.1.0",
        "libpython3.11.so.1.0",
        "libpython3.10.so.1.0",
        "libpython3.9.so.1.0",
        "libpython3.8.so.1.0",
        "python38.dll",
        "python39.dll",
        "python310.dll",
        "python311.dll",
        "python312.dll",
        "python313.dll",
        "python314.dll",
    }

    for name in runtime_names:
        if (root / name).exists():
            return True

    # Common PyInstaller support directories.
    support_names = {
        "PySide6",
        "PyQt5",
        "PyQt6",
        "numpy",
        "PIL",
        "lib-dynload",
    }

    for name in support_names:
        if (root / name).exists():
            return True

    return False


def _detect_onefile(
    executable: Path,
    archive: CArchive,
) -> bool:
    """
    Determine whether a PyInstaller executable is onefile.

    Important:
    The mere presence of sibling files does NOT automatically mean
    onedir. Arbitrary files can exist beside a onefile executable.

    We therefore look for the characteristic onedir runtime layout.
    """

    del archive

    return not _looks_like_onedir(
        executable
    )


# ---------------------------------------------------------------------------
# Safe paths
# ---------------------------------------------------------------------------


def _safe_output_path(
    root: Path,
    name: str,
) -> Path:
    """
    Resolve an archive entry safely.

    Prevents paths such as:

        ../../something

    from escaping the extraction directory.
    """

    # PyInstaller archive names use '/' regardless of host platform.
    normalized = name.replace(
        "\\",
        "/",
    )

    relative = Path(normalized)

    if relative.is_absolute():
        raise ExtractionError(
            f"Unsafe archive path: {name!r}"
        )

    if ".." in relative.parts:
        raise ExtractionError(
            f"Unsafe archive path: {name!r}"
        )

    root = root.resolve()
    destination = (
        root / relative
    ).resolve()

    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ExtractionError(
            f"Unsafe archive path: {name!r}"
        ) from error

    return destination


# ---------------------------------------------------------------------------
# CArchive extraction
# ---------------------------------------------------------------------------


def _classify_entry(
    entry_name: str,
    typecode: str,
) -> str:
    """
    Classify a raw CArchive entry.

    This does not modify the raw archive extraction.
    It only determines where a classified copy belongs.
    """

    lower = entry_name.lower()

    # Python archive.
    if lower.endswith(".pyz"):
        return "pyz"

    # Python standard library archive.
    if lower.endswith(".zip"):
        return "runtime"

    # Shared libraries / extensions.
    if lower.endswith(
        (
            ".so",
            ".so.1",
            ".dll",
            ".dylib",
            ".pyd",
        )
    ):
        return "libraries"

    # PyInstaller runtime/bootstrap.
    if (
        "pyiboot" in lower
        or "pyi_rth" in lower
        or "pyimod" in lower
    ):
        return "runtime"

    # Binary entries.
    if typecode in {
        "b",
        "d",
    }:
        return "libraries"

    return "data"


def _extract_carchive(
    archive: CArchive,
    output: Path,
) -> list[Path]:
    """
    Extract all supported CArchive entries.

    Everything is first written into:

        archive/raw/

    This is intentionally a faithful copy of the archive layer.
    """

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted: list[Path] = []

    for entry in archive.entries:
        if entry.typecode not in {
            "b",
            "x",
            "z",
            "s",
            "d",
        }:
            continue

        target = _safe_output_path(
            output,
            entry.name,
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            extracted_path = extract_entry(
                archive,
                entry,
                output,
            )
        except CArchiveError as error:
            raise ExtractionError(
                "Could not extract "
                f"{entry.name!r}: {error}"
            ) from error

        extracted.append(
            extracted_path
        )

    return extracted


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _copy_classified(
    raw_root: Path,
    output: Path,
    entries: list[dict],
) -> dict[str, int]:
    """
    Create a classified view of extracted CArchive files.

    Raw files remain untouched in archive/raw/.
    """

    counters = {
        "pyz": 0,
        "runtime": 0,
        "libraries": 0,
        "data": 0,
    }

    for item in entries:
        name = item["name"]
        typecode = item["type"]

        source = _safe_output_path(
            raw_root,
            name,
        )

        if not source.is_file():
            continue

        category = _classify_entry(
            name,
            typecode,
        )

        destination = _safe_output_path(
            output / category,
            name,
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        counters[category] += 1

    return counters


# ---------------------------------------------------------------------------
# TOC
# ---------------------------------------------------------------------------


def _toc_to_dict(
    archive: CArchive,
) -> list[dict]:
    return [
        {
            "name": entry.name,
            "type": entry.typecode,
            "position": entry.position,
            "compressed_size": (
                entry.compressed_size
            ),
            "uncompressed_size": (
                entry.uncompressed_size
            ),
            "compressed": entry.compressed,
        }
        for entry in archive.entries
    ]


def _write_toc(
    archive: CArchive,
    path: Path,
) -> list[dict]:
    """Write the CArchive TOC as JSON."""

    entries = _toc_to_dict(
        archive
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            entries,
            indent=2,
        ),
        encoding="utf-8",
    )

    return entries


# ---------------------------------------------------------------------------
# PYZ
# ---------------------------------------------------------------------------


def _find_pyz(
    raw_root: Path,
) -> Path | None:
    """Find the main PYZ archive."""

    candidates = [
        raw_root / "PYZ.pyz",
        *raw_root.glob(
            "*.pyz"
        ),
        *raw_root.glob(
            "*PYZ*.pyz"
        ),
    ]

    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in seen:
            continue

        seen.add(candidate)

        if candidate.is_file():
            return candidate

    return None


def _extract_pyz_layer(
    raw_root: Path,
    output: Path,
) -> dict:
    """
    Parse and extract the PYZ archive.

    Result:

        pyc/
            package/
                module.pyc

        pyz/
            toc.json
    """

    pyz_path = _find_pyz(
        raw_root
    )

    if pyz_path is None:
        return {
            "found": False,
            "entries": 0,
        }

    try:
        archive = parse_pyz(
            pyz_path
        )

        pyc_root = (
            output / "pyc"
        )

        extracted = extract_pyz(
            archive,
            pyc_root,
        )

        toc = [
            {
                "name": entry.name,
                "type": entry.typecode,
                "position": entry.position,
                "length": entry.length,
            }
            for entry in archive.entries
        ]

        pyz_root = (
            output / "pyz"
        )

        pyz_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            pyz_root / "toc.json"
        ).write_text(
            json.dumps(
                toc,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "found": True,
            "path": str(
                pyz_path
            ),
            "python_magic": (
                archive.python_magic.hex()
            ),
            "entries": len(
                extracted
            ),
        }

    except (
        PYZError,
        OSError,
    ) as error:
        return {
            "found": True,
            "path": str(
                pyz_path
            ),
            "entries": 0,
            "error": str(error),
        }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _python_version(
    encoded: int,
) -> str:
    major = encoded // 100
    minor = encoded % 100

    return f"{major}.{minor}"


def _write_metadata(
    bundle: BundleInfo,
    output: Path,
    *,
    toc_entries: list[dict],
    classified: dict[str, int] | None = None,
    pyz_info: dict | None = None,
) -> None:
    """Write complete extraction metadata."""

    python_version = _python_version(
        bundle.archive.python_version
    )

    metadata = {
        "tool": {
            "name": "pyarch",
        },
        "format": {
            "name": "pyinstaller",
            "version": "unknown",
        },
        "bundle": {
            "mode": (
                "onefile"
                if bundle.onefile
                else "onedir"
            ),
            "executable": str(
                bundle.executable
            ),
            "root": str(
                bundle.root
            ),
        },
        "python": {
            "version": python_version,
            "encoded_version": (
                bundle.archive.python_version
            ),
            "library": (
                bundle.archive.python_library
            ),
        },
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine(),
        },
        "archive": {
            "cookie_offset": (
                bundle.archive.cookie_offset
            ),
            "package_offset": (
                bundle.archive.package_offset
            ),
            "package_length": (
                bundle.archive.package_length
            ),
            "toc_offset": (
                bundle.archive.toc_offset
            ),
            "toc_length": (
                bundle.archive.toc_length
            ),
            "entries": len(
                toc_entries
            ),
        },
        "classification": (
            classified or {}
        ),
        "pyz": pyz_info,
    }

    (
        output / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# onedir
# ---------------------------------------------------------------------------


def _copy_onedir_bundle(
    bundle: BundleInfo,
    output: Path,
) -> None:
    """
    Copy a PyInstaller onedir bundle.

    The executable's parent directory is considered the bundle root.
    """

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_root = (
        bundle.root.resolve()
    )
    output_root = (
        output.resolve()
    )

    for item in source_root.iterdir():
        # Never copy the destination into itself.
        try:
            item.resolve().relative_to(
                output_root
            )
            continue
        except ValueError:
            pass

        destination = (
            output / item.name
        )

        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                symlinks=True,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(
                item,
                destination,
            )


# ---------------------------------------------------------------------------
# CArchive loading
# ---------------------------------------------------------------------------


def _load_carchive(
    executable: Path,
    archive: ArchiveInfo,
) -> CArchive:
    """Parse the CArchive TOC."""

    try:
        return parse_carchive(
            executable,
            package_offset=(
                archive.package_offset
            ),
            package_length=(
                archive.package_length
            ),
            toc_offset=(
                archive.toc_offset
            ),
            toc_length=(
                archive.toc_length
            ),
            python_version=(
                archive.python_version
            ),
            python_library=(
                archive.python_library
            ),
        )

    except (
        CArchiveError,
        OSError,
    ) as error:
        raise ExtractionError(
            "Could not parse PyInstaller "
            f"CArchive: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_bundle(
    path: Path,
) -> BundleInfo:
    """Detect and parse a PyInstaller executable."""

    path = path.resolve()

    if not path.is_file():
        raise ExtractionError(
            f"Executable not found: {path}"
        )

    archive = read_archive_info(
        path
    )

    carchive = _load_carchive(
        path,
        archive,
    )

    onefile = _detect_onefile(
        path,
        carchive,
    )

    return BundleInfo(
        executable=path,
        root=path.parent,
        onefile=onefile,
        archive=archive,
        carchive=carchive,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    path: Path,
    output: Path | None = None,
) -> BundleInfo:
    """
    Extract a PyInstaller application.

    Default:

        .pyarch/dist/<executable-name>/

    Onefile:

        executable/
        archive/raw/
        archive/toc.json
        pyz/
        pyc/
        runtime/
        libraries/
        data/
        metadata.json

    Onedir:

        original bundle layout
        metadata.json
    """

    bundle = _detect_bundle(
        path
    )

    if output is None:
        output = (
            Path(".pyarch")
            / "dist"
            / bundle.executable.name
        )
    else:
        output = output.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # ONEFILE
    # ---------------------------------------------------------------

    if bundle.onefile:
        if bundle.carchive is None:
            raise ExtractionError(
                "CArchive was not loaded."
            )

        archive_root = (
            output / "archive"
        )

        raw_root = (
            archive_root / "raw"
        )

        raw_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 1. Extract raw CArchive.
        _extract_carchive(
            bundle.carchive,
            raw_root,
        )

        # 2. Write CArchive TOC.
        toc_entries = _write_toc(
            bundle.carchive,
            archive_root / "toc.json",
        )

        # 3. Create classified view.
        classified = _copy_classified(
            raw_root,
            output,
            toc_entries,
        )

        # 4. Extract PYZ.
        pyz_info = _extract_pyz_layer(
            raw_root,
            output,
        )

        # 5. Preserve original executable.
        executable_dir = (
            output / "executable"
        )

        executable_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            bundle.executable,
            executable_dir
            / bundle.executable.name,
        )

        # 6. Metadata.
        _write_metadata(
            bundle,
            output,
            toc_entries=toc_entries,
            classified=classified,
            pyz_info=pyz_info,
        )

    # ---------------------------------------------------------------
    # ONEDIR
    # ---------------------------------------------------------------

    else:
        _copy_onedir_bundle(
            bundle,
            output,
        )

        toc_entries = _toc_to_dict(
            bundle.carchive
        ) if bundle.carchive else []

        pyz_info = None

        if bundle.carchive is not None:
            archive_root = (
                output / "archive"
            )

            raw_root = (
                archive_root / "raw"
            )

            raw_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            _write_toc(
                bundle.carchive,
                archive_root
                / "toc.json",
            )

            # Some onedir executables still contain a PYZ.
            pyz_info = _extract_pyz_layer(
                raw_root,
                output,
            )

        _write_metadata(
            bundle,
            output,
            toc_entries=toc_entries,
            classified=None,
            pyz_info=pyz_info,
        )

    return bundle