from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .pyc import load_pyc


@dataclass(slots=True)
class ModuleInfo:
    name: str
    path: str
    imports: list[str]
    functions: list[str]
    classes: list[str]


def module_name_from_path(
    root: Path,
    path: Path,
) -> str:
    relative = path.relative_to(root)

    parts = list(relative.parts)

    if parts[-1].endswith(".pyc"):
        parts[-1] = parts[-1][:-4]

    return ".".join(parts)


def analyze_pyc_tree(
    root: Path,
) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []

    for path in sorted(root.rglob("*.pyc")):
        try:
            pyc = load_pyc(path)
        except Exception:
            continue

        code = pyc.code

        imports: set[str] = set()
        functions: set[str] = set()
        classes: set[str] = set()

        for instruction in __import__(
            "dis"
        ).get_instructions(code):
            if instruction.opname in {
                "IMPORT_NAME",
            } and isinstance(
                instruction.argval,
                str,
            ):
                imports.add(
                    instruction.argval
                )

        for constant in code.co_consts:
            if isinstance(constant, str):
                continue

        functions.update(
            name
            for name in code.co_names
            if name.startswith("_")
            or name.isidentifier()
        )

        modules.append(
            ModuleInfo(
                name=module_name_from_path(
                    root,
                    path,
                ),
                path=str(path),
                imports=sorted(imports),
                functions=sorted(functions),
                classes=sorted(classes),
            )
        )

    return modules


def write_analysis(
    modules: list[ModuleInfo],
    output: Path,
) -> None:
    import json

    output.write_text(
        json.dumps(
            [
                {
                    "name": module.name,
                    "path": module.path,
                    "imports": module.imports,
                    "functions": module.functions,
                    "classes": module.classes,
                }
                for module in modules
            ],
            indent=2,
        ),
        encoding="utf-8",
    )