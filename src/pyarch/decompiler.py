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
    """
    Decompile a single .pyc file.

    Raises:
        RuntimeError: If the bytecode cannot be decoded.
    """

    try:
        pyc = load_pyc(path)
    except PYCError as error:
        raise RuntimeError(
            f"{path}: {error}"
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

    source/ contains CArchive application scripts.
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
    """
    Return every recovered .pyc file.

    The original PYZ extraction is kept under pyc/.
    """

    pyc_root = root / "pyc"

    if not pyc_root.is_dir():
        return []

    return sorted(
        pyc_root.rglob("*.pyc")
    )


def _relative_output_path(
    path: Path,
    root: Path,
    all_modules: bool,
) -> Path:
    """Calculate the output path for a bytecode file."""

    if all_modules:
        base = root / "pyc"
    elif path.is_relative_to(root / "source"):
        base = root / "source"
    else:
        base = root / "application"

    return path.relative_to(base)


def decompile_tree(
    root: Path,
    output: Path,
    *,
    all_modules: bool = False,
    progress: Progress | None = None,
    task_id: object | None = None,
) -> int:
    """
    Decompile recovered Python bytecode.

    By default only application-side code is processed.

    --all processes every recovered PYZ module, including stdlib.

    When a Rich Progress instance is supplied, the caller owns the
    progress task. Otherwise this function creates its own progress
    display.
    """

    root = root.resolve()
    output = output.resolve()

    inputs = (
        _all_inputs(root)
        if all_modules
        else _default_inputs(root)
    )

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

        task_id = progress.add_task(
            "Decompiling",
            total=len(inputs),
        )

    elif task_id is None:
        task_id = progress.add_task(
            "Decompiling",
            total=len(inputs),
        )

    assert progress is not None
    assert task_id is not None

    count = 0

    for path in inputs:
        relative = _relative_output_path(
            path,
            root,
            all_modules,
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

        progress.advance(
            task_id
        )

    if own_progress:
        progress.stop()

    return count
