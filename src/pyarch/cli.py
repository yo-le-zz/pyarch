from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ._metadata import (
    __description__,
    __package_name__,
    __version__,
)
from .classifier import classify_extraction
from .decompiler import decompile_tree
from .extractor import (
    ExtractionError,
    extract as extract_bundle,
)
from .inspector import write_inspection
from .reconstructor import reconstruct


app = typer.Typer(
    name=__package_name__,
    help=__description__,
    no_args_is_help=True,
)


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}"
        ),
        BarColumn(),
        TaskProgressColumn(),
    )


def version_callback(
    value: bool,
) -> None:
    if value:
        typer.echo(
            f"PyArch {__version__}"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show PyArch version and exit.",
    ),
) -> None:
    """Python binary reconstruction and analysis tool."""


@app.command()
def extract(
    path: Path,
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory.",
    ),
) -> None:
    """Extract a PyInstaller application."""

    with _progress() as progress:
        task = progress.add_task(
            "Extracting PyInstaller bundle",
            total=2,
        )

        try:
            bundle = extract_bundle(
                path,
                output,
            )
        except ExtractionError as error:
            typer.echo(
                f"Error: {error}",
                err=True,
            )
            raise typer.Exit(
                code=1
            ) from error

        progress.advance(task)

        destination = (
            output.resolve()
            if output is not None
            else (
                Path(".pyarch")
                / "dist"
                / bundle.executable.name
            )
        )

        classification = classify_extraction(
            destination
        )

        progress.advance(task)

    mode = (
        "onefile"
        if bundle.onefile
        else "onedir"
    )

    python_version = (
        f"{bundle.archive.python_version // 100}."
        f"{bundle.archive.python_version % 100}"
    )

    typer.echo()
    typer.echo(
        "✓ PyInstaller detected"
    )
    typer.echo(
        f"  Mode:       {mode}"
    )
    typer.echo(
        f"  Executable: {bundle.executable}"
    )
    typer.echo(
        f"  Python:     {python_version}"
    )
    typer.echo(
        f"  Output:     {destination}"
    )
    typer.echo()
    typer.echo(
        "  Recovery:"
    )
    typer.echo(
        f"    Source:       {classification['source']}"
    )
    typer.echo(
        f"    Application:  {classification['application']}"
    )
    typer.echo(
        f"    Stdlib:       {classification['stdlib']}"
    )
    typer.echo(
        f"    Runtime:      {classification['runtime']}"
    )


@app.command()
def decompile(
    path: Path,
    all_modules: bool = typer.Option(
        False,
        "--all",
        help=(
            "Decompile every recovered Python module, "
            "including stdlib."
        ),
    ),
) -> None:
    """Analyze and disassemble recovered Python bytecode."""

    root = path.resolve()

    if not (
        root / "source"
    ).is_dir():
        try:
            classify_extraction(root)
        except Exception as error:
            typer.echo(
                f"Warning: could not classify extraction: {error}",
                err=True,
            )

    output = root / "decompiled"

    with _progress() as progress:
        task = progress.add_task(
            "Decompiling",
            total=0,
        )

        count = decompile_tree(
            root,
            output,
            all_modules=all_modules,
            progress=progress,
            task_id=task,
        )

    typer.echo(
        f"✓ Processed {count} bytecode files"
    )
    typer.echo(
        "  Mode:   "
        f"{'all modules' if all_modules else 'application'}"
    )
    typer.echo(
        f"  Output: {output}"
    )


@app.command(
    "reconstruct"
)
def reconstruct_command(
    path: Path,
) -> None:
    """Reconstruct a Python project."""

    with _progress() as progress:
        task = progress.add_task(
            "Reconstructing project",
            total=1,
        )

        project = reconstruct(
            path.resolve()
        )

        progress.update(
            task,
            completed=1,
        )

    typer.echo(
        f"✓ Project reconstructed: {project}"
    )


@app.command()
def inspect(
    path: Path,
) -> None:
    """Inspect recovered files and architecture."""

    with _progress() as progress:
        task = progress.add_task(
            "Inspecting project",
            total=1,
        )

        output = write_inspection(
            path.resolve()
        )

        progress.update(
            task,
            completed=1,
        )

    typer.echo(
        f"✓ Inspection written to {output}"
    )


@app.command()
def make(
    path: Path,
) -> None:
    """Extract, decompile, reconstruct and inspect."""

    source = path.resolve()

    with _progress() as progress:
        # ---------------------------------------------------------
        # Extract
        # ---------------------------------------------------------

        extract_task = progress.add_task(
            "Extract",
            total=2,
        )

        try:
            bundle = extract_bundle(
                source,
                None,
            )
        except ExtractionError as error:
            typer.echo(
                f"Error: {error}",
                err=True,
            )
            raise typer.Exit(
                code=1
            ) from error

        progress.advance(
            extract_task
        )

        root = (
            Path(".pyarch")
            / "dist"
            / bundle.executable.name
        )

        classify_extraction(
            root
        )

        progress.advance(
            extract_task
        )

        # ---------------------------------------------------------
        # Decompile
        # ---------------------------------------------------------

        decompile_task = progress.add_task(
            "Decompile",
            total=0,
        )

        decompile_tree(
            root,
            root / "decompiled",
            progress=progress,
            task_id=decompile_task,
        )

        # ---------------------------------------------------------
        # Reconstruct
        # ---------------------------------------------------------

        reconstruct_task = progress.add_task(
            "Reconstruct",
            total=1,
        )

        reconstruct(
            root
        )

        progress.update(
            reconstruct_task,
            completed=1,
        )

        # ---------------------------------------------------------
        # Inspect
        # ---------------------------------------------------------

        inspect_task = progress.add_task(
            "Inspect",
            total=1,
        )

        write_inspection(
            root
        )

        progress.update(
            inspect_task,
            completed=1,
        )

    typer.echo()
    typer.echo(
        "✓ PyArch pipeline complete"
    )
    typer.echo(
        f"  Output: {root}"
    )


if __name__ == "__main__":
    app()
