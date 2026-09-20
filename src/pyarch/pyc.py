from __future__ import annotations

import dis
import marshal
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import BinaryIO

from .coderef import (
    RemoteCode,
    RemoteDisassemblyError,
    dump_remote,
    find_interpreter,
    is_code_object,
)
from .versions import (
    VersionDetectionError,
    detect_version,
)


class PYCError(Exception):
    """Raised when Python bytecode cannot be parsed."""


PYC_HEADER_SIZE = 16


@dataclass(slots=True)
class PYCFile:
    path: Path
    magic: bytes
    flags: int
    code: CodeType | RemoteCode
    python_version: tuple[int, int] | None = None
    used_remote_interpreter: bool = False


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
            target_version = detect_version(magic)
        except VersionDetectionError as error:
            raise PYCError(str(error)) from error

        host_version = sys.version_info[:2]

        if target_version == host_version:
            # Fast path: this interpreter's own `dis`/`marshal`
            # modules already agree with the bytecode's version.
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
                python_version=target_version,
                used_remote_interpreter=False,
            )

    # Version mismatch: disassembling in-process would use this
    # interpreter's opcode table on bytecode from a different
    # version, which can silently produce wrong instructions with no
    # error at all. Find a matching interpreter and let it do the
    # introspection correctly instead of guessing.
    interpreter = find_interpreter(*target_version)

    if interpreter is None:
        raise PYCError(
            f"{path} was compiled for Python "
            f"{target_version[0]}.{target_version[1]}, but PyArch is "
            f"running under Python {host_version[0]}.{host_version[1]} "
            f"and no matching 'python{target_version[0]}."
            f"{target_version[1]}' interpreter was found on PATH. "
            f"Install Python {target_version[0]}.{target_version[1]} "
            f"(it does not need to be your default interpreter) so "
            f"PyArch can disassemble this file correctly."
        )

    try:
        code = dump_remote(interpreter, path)
    except RemoteDisassemblyError as error:
        raise PYCError(
            f"Could not disassemble {path} using {interpreter}: "
            f"{error}"
        ) from error

    return PYCFile(
        path=path,
        magic=magic,
        flags=flags,
        code=code,
        python_version=target_version,
        used_remote_interpreter=True,
    )


def iter_code_objects(
    code: CodeType | RemoteCode,
):
    yield code

    for constant in code.co_consts:
        if is_code_object(constant):
            yield from iter_code_objects(constant)


def disassemble(
    code: CodeType | RemoteCode,
) -> str:
    lines: list[str] = []

    def walk(
        current: CodeType | RemoteCode,
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

        if isinstance(current, CodeType):
            for instruction in dis.Bytecode(current):
                lines.append(
                    f"{prefix}{instruction.offset:>4} "
                    f"{instruction.opname:<30} "
                    f"{instruction.argrepr}"
                )
        else:
            for instruction in current.pyarch_instructions:
                lines.append(
                    f"{prefix}{instruction.offset:>4} "
                    f"{instruction.op:<30} "
                    f"{instruction.argrepr}"
                )

        lines.append("")

        for constant in current.co_consts:
            if is_code_object(constant):
                walk(
                    constant,
                    depth + 1,
                )

    walk(code, 0)

    return "\n".join(lines)