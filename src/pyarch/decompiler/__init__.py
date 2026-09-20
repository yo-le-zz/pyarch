from __future__ import annotations

from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ..pyc import (
    PYCError,
    load_pyc,
)
from .engine import decompile_code


def decompile_file(
    path: Path,
    output: Path,
) -> list[str]:
    """
    Decompile a single .pyc file into Python source (.py).

    Returns diagnostic messages (warnings/errors) so callers can
    surface them without hiding failures.

    Raises:
        RuntimeError: If the bytecode cannot even be loaded.
    """

    try:
        pyc = load_pyc(path)
    except PYCError as error:
        raise RuntimeError(f"{path}: {error}") from error

    try:
        result = decompile_code(pyc.code)
    except Exception as error:
        # A single malformed/unusual code object must never abort an
        # entire batch (spec: "an unknown instruction must not
        # necessarily crash the whole decompilation"). Surface it
        # honestly in the output instead of raising.
        raise RuntimeError(
            f"unexpected error while decompiling {path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)

    source = result.source

    if not source:
        # Never invent source: if generation failed, say so instead
        # of writing an empty/misleading .py file.
        errors = "\n".join(f"# error: {m}" for m in result.diagnostics.errors)
        source = f"# PyArch could not reconstruct this module.\n#\n{errors}\n"

    output.write_text(source, encoding="utf-8")

    messages: list[str] = []
    messages.extend(f"warning: {m}" for m in result.diagnostics.warnings)
    messages.extend(f"error: {m}" for m in result.diagnostics.errors)
    return messages


def _default_inputs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in (root / "source", root / "application"):
        if directory.is_dir():
            paths.extend(directory.rglob("*.pyc"))
    return sorted(set(paths))


def _all_inputs(root: Path) -> list[Path]:
    pyc_root = root / "pyc"
    if not pyc_root.is_dir():
        return []
    return sorted(pyc_root.rglob("*.pyc"))


def _relative_output_path(path: Path, root: Path, all_modules: bool) -> Path:
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
    Decompile recovered Python bytecode into real .py source files.
    """

    root = root.resolve()
    output = output.resolve()

    inputs = _all_inputs(root) if all_modules else _default_inputs(root)

    own_progress = progress is None

    if own_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        progress.start()
        task_id = progress.add_task("Decompiling", total=len(inputs))
    elif task_id is None:
        task_id = progress.add_task("Decompiling", total=len(inputs))

    assert progress is not None
    assert task_id is not None

    count = 0

    for path in inputs:
        relative = _relative_output_path(path, root, all_modules)
        target = output / relative.with_suffix(".py")

        try:
            decompile_file(path, target)
        except RuntimeError as error:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                f"# PyArch could not decode this file:\n#\n# {error}\n",
                encoding="utf-8",
            )

        count += 1
        progress.advance(task_id)

    if own_progress:
        progress.stop()

    return count
