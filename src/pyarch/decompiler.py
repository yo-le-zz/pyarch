from __future__ import annotations

from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

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


def _default_inputs(
    root: Path,
) -> list[Path]:
    """
    Return application-side bytecode.

    source/ contains CArchive scripts.
    application/ contains non-stdlib PYZ modules.
    """

    paths: list[Path] = []

    for directory in (
        root / "source",
        root / "application",
    ):
        if not directory.is_dir():
            continue

        paths.extend(
            directory.rglob("*.pyc")
        )

    return sorted(
        set(paths)
    )


def _all_inputs(
    root: Path,
) -> list[Path]:
    pyc_root = (
        root / "pyc"
    )

    if not pyc_root.is_dir():
        return []

    return sorted(
        pyc_root.rglob("*.pyc")
    )


def decompile_tree(
    root: Path,
    output: Path,
    *,
    all_modules: bool = False,
    progress: Progress | None = None,
) -> int:
    """
    Decompile recovered Python bytecode.

    By default only application-side code is processed.

    --all processes every recovered PYZ module, including stdlib.
    """

    root = root.resolve()
    output = output.resolve()

    inputs = (
        _all_inputs(root)
        if all_modules
        else _default_inputs(root)
    )

    if not inputs:
        return 0

    own_progress = progress is None

    if own_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}"
            ),
            BarColumn(),
            TaskProgressColumn(),
        )
        progress.start()

    assert progress is not None

    task = progress.add_task(
        "Decompiling",
        total=len(inputs),
    )

    count = 0

    for path in inputs:
        if all_modules:
            relative = path.relative_to(
                root / "pyc"
            )
        else:
            if path.is_relative_to(
                root / "source"
            ):
                relative = path.relative_to(
                    root / "source"
                )
            else:
                relative = path.relative_to(
                    root / "application"
                )

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
                "PyArch could not decode this file:\n\n"
                f"{error}\n",
                encoding="utf-8",
            )

        count += 1

        progress.advance(task)

    if own_progress:
        progress.stop()

    return count