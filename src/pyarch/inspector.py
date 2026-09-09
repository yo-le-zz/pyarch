from __future__ import annotations

import json
from pathlib import Path


def inspect_project(
    root: Path,
) -> dict:
    result = {
        "root": str(root),
        "files": 0,
        "pyc": 0,
        "pyz": 0,
        "libraries": 0,
        "data": 0,
        "errors": [],
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        result["files"] += 1

        if path.suffix == ".pyc":
            result["pyc"] += 1

        elif path.suffix == ".pyz":
            result["pyz"] += 1

        elif path.suffix in {
            ".so",
            ".dll",
            ".dylib",
            ".pyd",
        }:
            result["libraries"] += 1

    return result


def write_inspection(
    root: Path,
) -> Path:
    result = inspect_project(root)

    output = root / "inspection.json"

    output.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output