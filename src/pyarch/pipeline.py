from pathlib import Path

from . import decompiler, extractor, inspector, reconstructor


def make(path: Path, output: Path) -> None:
    """Run the complete PyArch reconstruction pipeline."""

    extractor.extract(
        path,
        output,
    )

    decompiler.decompile(
        output,
        output,
    )

    reconstructor.reconstruct(
        output,
        output,
    )

    inspector.inspect(
        output,
    )