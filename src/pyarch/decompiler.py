from __future__ import annotations

import json
from pathlib import Path

from .pyc import (
    PYCError,
    disassemble,
    load_pyc,
)


def decompile_file(
    path: Path,
    output: Path,
) -> None:
    try:
        pyc = load_pyc(path)
    except PYCError as error:
        raise RuntimeError(
            str(error)
        ) from error

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        disassemble(pyc.code),
        encoding="utf-8",
    )


def decompile_tree(
    root: Path,
    output: Path,
) -> int:
    count = 0

    for path in sorted(
        root.rglob("*.pyc")
    ):
        relative = path.relative_to(root)

        target = (
            output
            / relative.with_suffix(
                ".dis.txt"
            )
        )

        try:
            decompile_file(
                path,
                target,
            )
        except RuntimeError as error:
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_text(
                f"PyArch could not decode this file:\n\n"
                f"{error}\n",
                encoding="utf-8",
            )

        count += 1

    return count