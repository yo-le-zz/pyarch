from __future__ import annotations

import json
from pathlib import Path


def reconstruct(
    root: Path,
) -> Path:
    project = root / "reconstructed"

    project.mkdir(
        parents=True,
        exist_ok=True,
    )

    pyc_root = root / "pyc"

    modules = []

    if pyc_root.exists():
        for path in sorted(
            pyc_root.rglob("*.pyc")
        ):
            modules.append(
                str(path.relative_to(root))
            )

    pyproject = project / "pyproject.toml"

    pyproject.write_text(
        """[project]
name = "reconstructed-project"
version = "0.0.0"
description = "Project reconstructed by PyArch."
requires-python = ">=3.13"
dependencies = []
""",
        encoding="utf-8",
    )

    (project / "pyarch.json").write_text(
        json.dumps(
            {
                "generated_by": "pyarch",
                "modules": modules,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return project