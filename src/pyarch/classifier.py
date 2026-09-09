from __future__ import annotations

import importlib.util
import json
import marshal
import shutil
import sys
from pathlib import Path

from .pyz import PYZError, parse_pyz


RUNTIME_NAMES = (
    "pyiboot",
    "pyi_rth",
    "pyimod",
)


def _is_runtime_name(name: str) -> bool:
    """Return whether a PyInstaller entry belongs to the runtime."""

    lower = name.lower()

    return any(
        marker in lower
        for marker in RUNTIME_NAMES
    )


def _safe_relative_path(name: str) -> Path:
    """Convert an archive name to a safe relative filesystem path."""

    normalized = name.replace(
        "\\",
        "/",
    )

    relative = Path(normalized)

    if relative.is_absolute():
        raise ValueError(
            f"Unsafe archive path: {name!r}"
        )

    parts = [
        part
        for part in relative.parts
        if part not in {"", "."}
    ]

    if ".." in parts:
        raise ValueError(
            f"Unsafe archive path: {name!r}"
        )

    if not parts:
        raise ValueError(
            f"Invalid archive entry name: {name!r}"
        )

    return Path(*parts)


def _copy_file(
    source: Path,
    target: Path,
) -> Path:
    """Copy a file while creating its parent directory."""

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        target,
    )

    return target


def _get_python_magic(
    root: Path,
) -> bytes:
    """
    Determine the Python magic number used by the bundle.

    The PYZ header is the best source because it contains the exact
    magic used to marshal the bytecode contained in the archive.

    importlib.util.MAGIC_NUMBER is used as a fallback.
    """

    raw_root = (
        root
        / "archive"
        / "raw"
    )

    candidates = [
        raw_root / "PYZ.pyz",
        *raw_root.glob("*.pyz"),
        *raw_root.glob("*PYZ*.pyz"),
    ]

    seen: set[Path] = set()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue

        if candidate in seen:
            continue

        seen.add(candidate)

        if not candidate.is_file():
            continue

        try:
            archive = parse_pyz(candidate)

            magic = archive.python_magic

            if len(magic) == 4:
                return magic

        except (
            PYZError,
            OSError,
        ):
            continue

    return importlib.util.MAGIC_NUMBER


def _make_pyc(
    magic: bytes,
    payload: bytes,
) -> bytes:
    """
    Build a valid Python 3.7+ .pyc file.

    PyInstaller CArchive ``s`` entries contain the marshalled
    code object without the normal .pyc header.

    Header layout:

        4 bytes  magic
        4 bytes  flags
        8 bytes  timestamp/hash data
    """

    if len(magic) != 4:
        raise ValueError(
            "Python magic must contain exactly 4 bytes."
        )

    # Validate the payload before writing it.
    try:
        code = marshal.loads(payload)
    except Exception as error:
        raise ValueError(
            "Source payload does not contain a valid "
            "marshalled Python object."
        ) from error

    # The source entry should contain a code object.
    if not hasattr(code, "co_code"):
        raise ValueError(
            "Source payload does not contain a Python code object."
        )

    return (
        magic
        + b"\x00\x00\x00\x00"
        + b"\x00" * 8
        + payload
    )


def _copy_source(
    source: Path,
    output: Path,
    name: str,
    magic: bytes,
) -> Path:
    """
    Convert a PyInstaller ``s`` payload into a real .pyc file.

    The raw CArchive payload is never modified.
    """

    relative = _safe_relative_path(
        name
    )

    if relative.suffix != ".pyc":
        relative = relative.with_suffix(
            ".pyc"
        )

    target = (
        output
        / "source"
        / relative
    )

    payload = source.read_bytes()

    pyc = _make_pyc(
        magic,
        payload,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_bytes(
        pyc
    )

    return target


def _copy_category(
    source: Path,
    output: Path,
    category: str,
    name: str,
) -> Path:
    """Copy an extracted archive entry into a category directory."""

    relative = _safe_relative_path(
        name
    )

    target = (
        output
        / category
        / relative
    )

    return _copy_file(
        source,
        target,
    )


def _classify_carchive_entry(
    name: str,
    typecode: str,
) -> str:
    """Classify a PyInstaller CArchive entry."""

    lower = name.lower()

    if lower.endswith(".pyz"):
        return "pyz"

    if lower.endswith(".zip"):
        return "runtime"

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

    if _is_runtime_name(name):
        return "runtime"

    if typecode in {
        "b",
        "d",
    }:
        return "libraries"

    # ``s`` is a Python script payload.
    if typecode == "s":
        return "source"

    return "data"


def _classify_pyz_module(
    name: str,
) -> str:
    """
    Classify a PYZ module as stdlib or application.

    Python's stdlib module list is used for the first-level
    classification. Third-party dependencies are therefore grouped
    with application-side modules for now.
    """

    normalized = name.replace(
        "\\",
        "/",
    )

    if "/" in normalized:
        top_level = normalized.split(
            "/",
            1,
        )[0]
    else:
        top_level = normalized.split(
            ".",
            1,
        )[0]

    stdlib = getattr(
        sys,
        "stdlib_module_names",
        set(),
    )

    if top_level in stdlib:
        return "stdlib"

    return "application"


def _load_json(
    path: Path,
) -> object:
    """Load a JSON file."""

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _load_carchive_toc(
    root: Path,
) -> list[dict]:
    """Load the extracted CArchive TOC."""

    toc_path = (
        root
        / "archive"
        / "toc.json"
    )

    if not toc_path.is_file():
        return []

    try:
        data = _load_json(
            toc_path
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []

    if not isinstance(data, list):
        return []

    return [
        entry
        for entry in data
        if isinstance(entry, dict)
    ]


def _load_pyz_toc(
    root: Path,
) -> list[dict]:
    """Load the extracted PYZ TOC."""

    toc_path = (
        root
        / "pyz"
        / "toc.json"
    )

    if not toc_path.is_file():
        return []

    try:
        data = _load_json(
            toc_path
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return []

    if not isinstance(data, list):
        return []

    return [
        entry
        for entry in data
        if isinstance(entry, dict)
    ]


def _classify_carchive(
    root: Path,
    output: Path,
    counters: dict[str, int],
) -> None:
    """Classify files extracted from the CArchive."""

    raw_root = (
        root
        / "archive"
        / "raw"
    )

    if not raw_root.is_dir():
        return

    magic = _get_python_magic(
        root
    )

    for entry in _load_carchive_toc(
        root
    ):
        name = entry.get("name")
        typecode = entry.get("type")

        if not isinstance(name, str):
            continue

        if not isinstance(typecode, str):
            continue

        try:
            relative = _safe_relative_path(
                name
            )
        except ValueError:
            continue

        source = (
            raw_root
            / relative
        )

        if not source.is_file():
            continue

        category = _classify_carchive_entry(
            name,
            typecode,
        )

        if category == "source":
            try:
                _copy_source(
                    source,
                    output,
                    name,
                    magic,
                )
            except (
                OSError,
                ValueError,
            ):
                # Keep the raw entry available even if reconstruction
                # of this particular source payload fails.
                continue

        elif category == "pyz":
            _copy_category(
                source,
                output,
                "pyz",
                name,
            )

        else:
            _copy_category(
                source,
                output,
                category,
                name,
            )

        counters[category] += 1


def _build_module_map(
    pyc_root: Path,
) -> dict[str, Path]:
    """Build a mapping from PYZ module names to extracted .pyc files."""

    module_paths: dict[str, Path] = {}

    for path in sorted(
        pyc_root.rglob("*.pyc")
    ):
        relative = path.relative_to(
            pyc_root
        )

        module_name = ".".join(
            relative.with_suffix("").parts
        )

        module_paths[module_name] = path

    return module_paths


def _classify_pyz(
    root: Path,
    output: Path,
    counters: dict[str, int],
) -> None:
    """Classify modules extracted from PYZ."""

    pyc_root = (
        root
        / "pyc"
    )

    if not pyc_root.is_dir():
        return

    module_paths = _build_module_map(
        pyc_root
    )

    toc = _load_pyz_toc(
        root
    )

    processed: set[Path] = set()

    if toc:
        for entry in toc:
            name = entry.get("name")

            if not isinstance(name, str):
                continue

            normalized = (
                name.replace("/", ".")
                .replace("\\", ".")
            )

            source = module_paths.get(
                name
            )

            if source is None:
                source = module_paths.get(
                    normalized
                )

            if source is None:
                continue

            if source in processed:
                continue

            processed.add(source)

            category = _classify_pyz_module(
                name
            )

            relative = source.relative_to(
                pyc_root
            )

            target = (
                output
                / category
                / relative
            )

            _copy_file(
                source,
                target,
            )

            counters[category] += 1

    # Process any extracted .pyc that wasn't represented by the TOC.
    for source in sorted(
        pyc_root.rglob("*.pyc")
    ):
        if source in processed:
            continue

        relative = source.relative_to(
            pyc_root
        )

        module_name = ".".join(
            relative.with_suffix("").parts
        )

        category = _classify_pyz_module(
            module_name
        )

        target = (
            output
            / category
            / relative
        )

        _copy_file(
            source,
            target,
        )

        counters[category] += 1


def _update_metadata(
    root: Path,
    counters: dict[str, int],
) -> None:
    """Add classification information to metadata.json."""

    metadata_path = (
        root
        / "metadata.json"
    )

    if metadata_path.is_file():
        try:
            metadata = _load_json(
                metadata_path
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            metadata = {}
    else:
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}

    metadata["recovery"] = {
        "source": counters["source"],
        "application": counters["application"],
        "stdlib": counters["stdlib"],
        "runtime": counters["runtime"],
        "libraries": counters["libraries"],
        "data": counters["data"],
        "pyz": counters["pyz"],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def classify_extraction(
    root: Path,
) -> dict[str, int]:
    """
    Classify an extracted PyInstaller project.

    Raw archive files and the original PYZ-extracted bytecode are
    preserved. Classified copies are placed under:

        source/
        application/
        stdlib/
        runtime/
        libraries/
        data/
        pyz/
    """

    root = root.resolve()

    counters = {
        "source": 0,
        "application": 0,
        "stdlib": 0,
        "runtime": 0,
        "libraries": 0,
        "data": 0,
        "pyz": 0,
    }

    if not root.is_dir():
        return counters

    # Remove the previous classification.
    for directory in (
        "source",
        "application",
        "stdlib",
        "runtime",
        "libraries",
        "data",
        "pyz",
    ):
        path = root / directory

        if path.exists():
            shutil.rmtree(
                path
            )

    _classify_carchive(
        root,
        root,
        counters,
    )

    _classify_pyz(
        root,
        root,
        counters,
    )

    _update_metadata(
        root,
        counters,
    )

    return counters
