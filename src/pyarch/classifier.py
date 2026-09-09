from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


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

    normalized = name.replace("\\", "/")

    parts = [
        part
        for part in normalized.split("/")
        if part not in {"", ".", ".."}
    ]

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


def _copy_source(
    source: Path,
    output: Path,
    name: str,
) -> Path:
    """
    Copy a PyInstaller script payload into source/.

    PyInstaller CArchive entries of type ``s`` contain the
    marshalled Python bytecode directly and normally have no
    .pyc extension. PyArch exposes them as .pyc files so that
    the rest of the pipeline can process them normally.
    """

    relative = _safe_relative_path(name)

    if relative.suffix != ".pyc":
        relative = relative.with_suffix(
            relative.suffix + ".pyc"
            if relative.suffix
            else ".pyc"
        )

    target = (
        output
        / "source"
        / relative
    )

    return _copy_file(
        source,
        target,
    )


def _copy_category(
    source: Path,
    output: Path,
    category: str,
    name: str,
) -> Path:
    """Copy an extracted archive entry into a category directory."""

    relative = _safe_relative_path(name)

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

    # PYZ archives.
    if lower.endswith(".pyz"):
        return "pyz"

    # Python's bundled standard library archive.
    if lower.endswith(".zip"):
        return "stdlib"

    # Native libraries.
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

    # PyInstaller bootloader/runtime modules.
    if _is_runtime_name(name):
        return "runtime"

    # PyInstaller binary/data entries.
    if typecode in {
        "b",
        "d",
    }:
        return "libraries"

    # ``s`` means a Python script/source payload.
    #
    # This is where the application's main.pyc normally lives.
    if typecode == "s":
        return "source"

    # Generic binary/data.
    return "data"


def _classify_pyz_module(
    name: str,
) -> str:
    """
    Classify a PYZ module as stdlib or application.

    PyInstaller does not provide enough information in the PYZ
    TOC to perfectly distinguish third-party packages from
    application code. The top-level module name is therefore
    compared against Python's standard-library module list.

    Non-stdlib modules are considered application-side for now.
    """

    normalized = name.replace("\\", "/")

    if "/" in normalized:
        top_level = normalized.split("/", 1)[0]
    else:
        top_level = normalized.split(".", 1)[0]

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

    data = _load_json(toc_path)

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

    data = _load_json(toc_path)

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
    """Classify files extracted from the PyInstaller CArchive."""

    raw_root = (
        root
        / "archive"
        / "raw"
    )

    if not raw_root.is_dir():
        return

    for entry in _load_carchive_toc(root):
        name = entry.get("name")
        typecode = entry.get("type")

        if not isinstance(name, str):
            continue

        if not isinstance(typecode, str):
            continue

        source = (
            raw_root
            / _safe_relative_path(name)
        )

        if not source.is_file():
            continue

        category = _classify_carchive_entry(
            name,
            typecode,
        )

        # PYZ is handled separately by the PYZ classifier.
        if category == "pyz":
            target = (
                output
                / "pyz"
                / source.name
            )

            _copy_file(
                source,
                target,
            )

            counters["pyz"] += 1
            continue

        if category == "source":
            _copy_source(
                source,
                output,
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

    toc = _load_pyz_toc(root)

    # Build a mapping from module name to its extracted .pyc file.
    module_paths: dict[str, Path] = {}

    for path in sorted(
        pyc_root.rglob("*.pyc")
    ):
        relative = path.relative_to(
            pyc_root
        )

        # Convert:
        #     package/module.pyc
        # into:
        #     package.module
        module_name = ".".join(
            relative.with_suffix("").parts
        )

        module_paths[module_name] = path

    # Prefer the actual PYZ TOC when available.
    if toc:
        for entry in toc:
            name = entry.get("name")

            if not isinstance(name, str):
                continue

            source = module_paths.get(name)

            if source is None:
                # Some PYZ names can use slightly different
                # representations. Try the normalized path.
                normalized = (
                    name.replace("/", ".")
                    .replace("\\", ".")
                )

                source = module_paths.get(
                    normalized
                )

            if source is None:
                continue

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

        return

    # Fallback when pyz/toc.json is unavailable.
    for source in sorted(
        pyc_root.rglob("*.pyc")
    ):
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

    The original archive/raw and pyc directories are preserved.
    Classified copies are created under:

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

    # Remove previous classification output.
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
            shutil.rmtree(path)

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