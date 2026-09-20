from __future__ import annotations

"""
Cross-version code object support.

When a .pyc's magic number doesn't match the Python interpreter that
is running PyArch, disassembling it in-process would silently use
the WRONG opcode table (the host interpreter's), producing incorrect
bytecode with no error. To decompile it correctly PyArch shells out
to a matching Python interpreter (found on PATH) and asks it to dump
the code object's structure as JSON -- using that interpreter's own,
correct `dis`/`marshal` modules. No code from the analyzed program is
ever executed; only introspected.

The result is wrapped in `RemoteCode`, a plain object exposing the
same attribute names as `types.CodeType` (`co_name`, `co_consts`,
etc.) so the rest of PyArch can use it without caring whether it came
from the local interpreter or a remote one -- except for a handful of
`isinstance(x, CodeType)` checks, which must use `is_code_object(x)`
instead so they also accept `RemoteCode`.
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType


class RemoteDisassemblyError(Exception):
    """Raised when a matching interpreter could not disassemble a .pyc."""


@dataclass(slots=True)
class RemoteInstruction:
    """One instruction, decoded by the matching interpreter."""

    offset: int
    op: str
    opcode: int
    arg: int | None
    argval: object
    argrepr: str

    @property
    def opname(self) -> str:
        return self.op


@dataclass(slots=True)
class RemoteCode:
    """
    A `types.CodeType` stand-in built from another interpreter's
    introspection of a .pyc this interpreter cannot safely decode
    itself.
    """

    co_name: str = ""
    co_qualname: str = ""
    co_filename: str = ""
    co_firstlineno: int = 0
    co_argcount: int = 0
    co_posonlyargcount: int = 0
    co_kwonlyargcount: int = 0
    co_nlocals: int = 0
    co_flags: int = 0
    co_varnames: tuple = ()
    co_names: tuple = ()
    co_freevars: tuple = ()
    co_cellvars: tuple = ()
    co_consts: tuple = ()
    co_exceptiontable: bytes = b""

    # Pre-decoded instructions, computed by the matching interpreter
    # using its own, correct opcode table. `translate.py` uses these
    # directly instead of calling `dis.get_instructions()`.
    pyarch_instructions: list[RemoteInstruction] = field(
        default_factory=list
    )

    pyarch_version: tuple[int, int] = (0, 0)


def is_code_object(value: object) -> bool:
    """True for a real `types.CodeType` or a `RemoteCode`."""

    return isinstance(value, (CodeType, RemoteCode))


# ---------------------------------------------------------------------------
# Interpreter discovery
# ---------------------------------------------------------------------------


def host_version() -> tuple[int, int]:
    return sys.version_info[:2]


def find_interpreter(major: int, minor: int) -> str | None:
    """
    Look for a Python interpreter matching `major.minor` on PATH.

    Tries the conventional Unix naming first (``python3.13``), then a
    couple of common alternatives. Returns an absolute path, or None
    if nothing matching was found.
    """

    candidates = [
        f"python{major}.{minor}",
        f"python{major}{minor}",
    ]

    for name in candidates:
        found = shutil.which(name)

        if found:
            return found

    return None


# ---------------------------------------------------------------------------
# Remote dump
# ---------------------------------------------------------------------------

# Runs under the TARGET interpreter. Reads the .pyc, marshal-loads
# the (already version-appropriate) code object using that
# interpreter's own marshal/dis modules, and prints a JSON dump of
# its structure. Never executes anything from the analyzed program.
_DUMP_SCRIPT = r"""
import dis, json, marshal, sys
from types import CodeType

# Code objects are referenced from two places in the dump: their
# owning parent's co_consts, and every LOAD_CONST instruction that
# pushes them. JSON has no shared-reference concept like Python
# does, so encoding a code object inline at both places would create
# two structurally-identical but separately-decoded copies -- and
# PyArch matches a MAKE_FUNCTION's code constant back to its
# already-decompiled body by Python object identity. Encoding every
# code object exactly once, in a flat table, keyed by id(), and
# referencing it everywhere else by that id, keeps one Python object
# per code object after decoding, matching what happens naturally
# with a real (non-remote) CodeType.
code_table = {}

def encode_value(value):
    if isinstance(value, CodeType):
        ref = id(value)
        if ref not in code_table:
            code_table[ref] = None  # reserve, avoid infinite recursion
            code_table[ref] = encode_code(value)
        return {"$coderef": ref}
    if isinstance(value, tuple):
        return {"$tuple": [encode_value(v) for v in value]}
    if isinstance(value, frozenset):
        return {"$frozenset": [encode_value(v) for v in value]}
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, complex):
        return {"$complex": [value.real, value.imag]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"$repr": repr(value)}

def encode_code(code):
    instructions = []
    for instr in dis.get_instructions(code):
        instructions.append({
            "offset": instr.offset,
            "op": instr.opname,
            "opcode": instr.opcode,
            "arg": instr.arg,
            "argval": encode_value(instr.argval),
            "argrepr": instr.argrepr,
        })
    return {
        "co_name": code.co_name,
        "co_qualname": getattr(code, "co_qualname", code.co_name),
        "co_filename": code.co_filename,
        "co_firstlineno": code.co_firstlineno,
        "co_argcount": code.co_argcount,
        "co_posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_flags": code.co_flags,
        "co_varnames": list(code.co_varnames),
        "co_names": list(code.co_names),
        "co_freevars": list(code.co_freevars),
        "co_cellvars": list(code.co_cellvars),
        "co_consts": [encode_value(c) for c in code.co_consts],
        "co_exceptiontable": getattr(
            code, "co_exceptiontable", b""
        ).hex(),
        "instructions": instructions,
    }

with open(sys.argv[1], "rb") as f:
    header = f.read(16)
    code = marshal.load(f)

if not isinstance(code, CodeType):
    print(json.dumps({"error": "not a code object"}))
    sys.exit(1)

root_ref = id(code)
code_table[root_ref] = encode_code(code)

print(json.dumps({
    "version": list(sys.version_info[:2]),
    "root": root_ref,
    "code_table": code_table,
}))
"""


def _decode_value(
    value: object,
    decoded: dict[int, "RemoteCode"],
    table: dict,
) -> object:
    if isinstance(value, dict):
        if "$coderef" in value:
            return _decode_code(value["$coderef"], decoded, table)
        if "$tuple" in value:
            return tuple(
                _decode_value(v, decoded, table)
                for v in value["$tuple"]
            )
        if "$frozenset" in value:
            return frozenset(
                _decode_value(v, decoded, table)
                for v in value["$frozenset"]
            )
        if "$bytes" in value:
            return bytes.fromhex(value["$bytes"])
        if "$complex" in value:
            real, imag = value["$complex"]
            return complex(real, imag)
        if "$repr" in value:
            # Opaque constant PyArch cannot faithfully reconstruct
            # (e.g. an exotic marshalled type). Kept as a marker
            # string rather than guessed at.
            return value["$repr"]

    return value


def _decode_code(
    ref: int,
    decoded: dict[int, "RemoteCode"],
    table: dict,
) -> "RemoteCode":
    """
    Decode the code object identified by `ref`, memoized so every
    reference to the same original Python code object resolves to
    the exact same `RemoteCode` instance -- required for PyArch's
    identity-based `code_map` lookups to work.
    """

    existing = decoded.get(ref)

    if existing is not None:
        return existing

    data = table[str(ref)] if str(ref) in table else table[ref]

    # Reserve a placeholder before recursing into co_consts, in case
    # of (unexpected) self-reference.
    placeholder = RemoteCode()
    decoded[ref] = placeholder

    instructions = [
        RemoteInstruction(
            offset=item["offset"],
            op=item["op"],
            opcode=item["opcode"],
            arg=item["arg"],
            argval=_decode_value(item["argval"], decoded, table),
            argrepr=item["argrepr"],
        )
        for item in data["instructions"]
    ]

    placeholder.co_name = data["co_name"]
    placeholder.co_qualname = data["co_qualname"]
    placeholder.co_filename = data["co_filename"]
    placeholder.co_firstlineno = data["co_firstlineno"]
    placeholder.co_argcount = data["co_argcount"]
    placeholder.co_posonlyargcount = data["co_posonlyargcount"]
    placeholder.co_kwonlyargcount = data["co_kwonlyargcount"]
    placeholder.co_nlocals = data["co_nlocals"]
    placeholder.co_flags = data["co_flags"]
    placeholder.co_varnames = tuple(data["co_varnames"])
    placeholder.co_names = tuple(data["co_names"])
    placeholder.co_freevars = tuple(data["co_freevars"])
    placeholder.co_cellvars = tuple(data["co_cellvars"])
    placeholder.co_consts = tuple(
        _decode_value(c, decoded, table) for c in data["co_consts"]
    )
    placeholder.co_exceptiontable = bytes.fromhex(
        data.get("co_exceptiontable", "")
    )
    placeholder.pyarch_instructions = instructions

    return placeholder


def dump_remote(interpreter: str, pyc_path: Path) -> RemoteCode:
    """
    Ask `interpreter` to introspect `pyc_path` and return the result
    as a `RemoteCode` tree (nested code objects included via
    `co_consts`, exactly like a real `CodeType`), preserving object
    identity for repeated references the same way a real `marshal`
    load would.
    """

    try:
        result = subprocess.run(
            [interpreter, "-c", _DUMP_SCRIPT, str(pyc_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RemoteDisassemblyError(
            f"Could not run {interpreter}: {error}"
        ) from error

    if result.returncode != 0:
        raise RemoteDisassemblyError(
            f"{interpreter} failed to introspect {pyc_path}: "
            f"{result.stderr.strip()}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RemoteDisassemblyError(
            f"{interpreter} produced no usable output for "
            f"{pyc_path}: {error}"
        ) from error

    if "error" in payload:
        raise RemoteDisassemblyError(
            f"{interpreter}: {payload['error']}"
        )

    decoded: dict[int, RemoteCode] = {}
    table = payload["code_table"]
    code = _decode_code(payload["root"], decoded, table)
    code.pyarch_version = tuple(payload["version"])

    return code
