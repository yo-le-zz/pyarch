from __future__ import annotations

import dis
import marshal
import struct
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import BinaryIO


class PYCError(Exception):
    """Raised when Python bytecode cannot be parsed."""


PYC_HEADER_SIZE = 16


@dataclass(slots=True)
class PYCFile:
    path: Path
    magic: bytes
    flags: int
    code: CodeType


def _read_exact(
    file: BinaryIO,
    size: int,
) -> bytes:
    data = file.read(size)

    if len(data) != size:
        raise PYCError(
            "Unexpected end of .pyc file."
        )

    return data


def load_pyc(path: Path) -> PYCFile:
    path = path.resolve()

    if not path.is_file():
        raise PYCError(
            f"Bytecode file not found: {path}"
        )

    with path.open("rb") as file:
        header = _read_exact(
            file,
            PYC_HEADER_SIZE,
        )

        magic = header[:4]
        flags = struct.unpack(
            "<I",
            header[4:8],
        )[0]

        try:
            code = marshal.load(file)
        except Exception as error:
            raise PYCError(
                f"Could not load bytecode from {path}."
            ) from error

    if not isinstance(code, CodeType):
        raise PYCError(
            f"{path} does not contain a code object."
        )

    return PYCFile(
        path=path,
        magic=magic,
        flags=flags,
        code=code,
    )


def iter_code_objects(
    code: CodeType,
):
    yield code

    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from iter_code_objects(constant)


def disassemble(
    code: CodeType,
) -> str:
    lines: list[str] = []

    def walk(
        current: CodeType,
        depth: int,
    ) -> None:
        prefix = "    " * depth

        lines.append(
            f"{prefix}Code object: "
            f"{current.co_name}"
        )

        lines.append(
            f"{prefix}File: "
            f"{current.co_filename}"
        )

        lines.append(
            f"{prefix}Arguments: "
            f"{current.co_argcount}"
        )

        lines.append(
            f"{prefix}Locals: "
            f"{current.co_nlocals}"
        )

        lines.append("")

        output = dis.Bytecode(current)

        for instruction in output:
            lines.append(
                f"{prefix}{instruction.offset:>4} "
                f"{instruction.opname:<30} "
                f"{instruction.argrepr}"
            )

        lines.append("")

        for constant in current.co_consts:
            if isinstance(constant, CodeType):
                walk(
                    constant,
                    depth + 1,
                )

    walk(code, 0)

    return "\n".join(lines)