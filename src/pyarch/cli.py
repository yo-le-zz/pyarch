from pathlib import Path

import typer

from ._metadata import (
    __description__,
    __package_name__,
    __version__,
)
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


def version_callback(value: bool) -> None:
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
        raise typer.Exit(code=1) from error

    mode = (
        "onefile"
        if bundle.onefile
        else "onedir"
    )

    destination = (
        output.resolve()
        if output is not None
        else (
            Path(".pyarch")
            / "dist"
            / bundle.executable.name
        )
    )

    python_version = (
        f"{bundle.archive.python_version // 100}."
        f"{bundle.archive.python_version % 100}"
    )

    typer.echo("✓ PyInstaller detected")
    typer.echo(f"  Mode:       {mode}")
    typer.echo(f"  Executable: {bundle.executable}")
    typer.echo(f"  Python:     {python_version}")
    typer.echo(f"  Output:     {destination}")


@app.command()
def decompile(
    path: Path,
) -> None:
    """Analyze and disassemble recovered Python bytecode."""

    root = path.resolve()

    pyc_root = (
        root / "pyc"
        if (root / "pyc").is_dir()
        else root
    )

    output = root / "decompiled"

    count = decompile_tree(
        pyc_root,
        output,
    )

    typer.echo(
        f"✓ Processed {count} bytecode files"
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
    project = reconstruct(
        path.resolve()
    )

    typer.echo(
        f"✓ Project reconstructed: {project}"
    )


@app.command()
def inspect(
    path: Path,
) -> None:
    """Inspect recovered files and architecture."""

    output = write_inspection(
        path.resolve()
    )

    typer.echo(
        f"✓ Inspection written to {output}"
    )


@app.command()
def make(
    path: Path,
) -> None:
    """Extract, analyze, reconstruct and inspect."""

    source = path.resolve()

    typer.echo(
        "╭─ PyArch"
    )
    typer.echo(
        "│ Extracting..."
    )

    try:
        bundle = extract_bundle(
            source,
            None,
        )
    except ExtractionError as error:
        typer.echo(
            f"│ Error: {error}",
            err=True,
        )
        raise typer.Exit(
            code=1
        ) from error

    root = (
        Path(".pyarch")
        / "dist"
        / bundle.executable.name
    )

    typer.echo(
        "│ ✓ Extraction complete"
    )

    typer.echo(
        "│ Decompiling..."
    )

    decompile_tree(
        root / "pyc",
        root / "decompiled",
    )

    typer.echo(
        "│ ✓ Bytecode analyzed"
    )

    typer.echo(
        "│ Reconstructing..."
    )

    reconstruct(root)

    typer.echo(
        "│ ✓ Project reconstructed"
    )

    typer.echo(
        "│ Inspecting..."
    )

    write_inspection(root)

    typer.echo(
        "│ ✓ Inspection complete"
    )
    typer.echo(
        "╰─ Done."
    )


if __name__ == "__main__":
    app()