from __future__ import annotations

"""
CPython bytecode normalization layer.

The goal of this module is NOT to decompile bytecode.

It translates CPython-version-specific instructions into a small,
version-independent instruction vocabulary that the real decompiler
can consume.

Pipeline:

    CPython bytecode
          ↓
    dis.Instruction
          ↓
    TInstruction
          ↓
    version-independent decompiler
"""


import dis
import sys

from dataclasses import dataclass
from types import CodeType
from typing import Any


@dataclass(slots=True, frozen=True)
class TInstruction:
    """
    A normalized bytecode instruction.

    `op` is a semantic operation understood by PyArch.

    `original` contains the original CPython opcode name so that
    diagnostics can still tell us what CPython actually emitted.
    """

    offset: int
    op: str

    arg: int | None = None
    argval: Any = None
    argrepr: str = ""

    target: int | None = None

    original: str = ""


# ---------------------------------------------------------------------------
# Semantic instruction vocabulary
# ---------------------------------------------------------------------------

SEMANTIC_OPS = frozenset(
    {
        # Execution / no-op
        "RESUME",
        "NOP",

        # Constants / names
        "LOAD_CONST",
        "LOAD_NAME",
        "STORE_NAME",
        "DELETE_NAME",

        "LOAD_FAST",
        "STORE_FAST",
        "DELETE_FAST",

        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",

        "LOAD_DEREF",
        "STORE_DEREF",
        "DELETE_DEREF",

        # Attributes
        "LOAD_ATTR",
        "STORE_ATTR",
        "DELETE_ATTR",

        "LOAD_METHOD",
        "LOAD_SUPER_ATTR",

        # Calls
        "PUSH_NULL",
        "PRECALL",
        "CALL",
        "CALL_FUNCTION_EX",
        "KW_NAMES",

        # Stack
        "POP_TOP",
        "COPY",
        "SWAP",

        # Operators
        "BINARY_OP",
        "COMPARE_OP",
        "IS_OP",
        "CONTAINS_OP",

        "UNARY_NOT",
        "UNARY_NEGATIVE",
        "UNARY_POSITIVE",
        "UNARY_INVERT",

        # Containers
        "BUILD_LIST",
        "BUILD_TUPLE",
        "BUILD_SET",
        "BUILD_MAP",
        "BUILD_CONST_KEY_MAP",

        "LIST_EXTEND",
        "SET_UPDATE",
        "DICT_UPDATE",

        # Indexing
        "BINARY_SUBSCR",
        "STORE_SUBSCR",
        "DELETE_SUBSCR",

        # Control flow
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",

        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",

        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",

        "FOR_ITER",
        "GET_ITER",
        "END_FOR",

        # Sequence operations
        "UNPACK_SEQUENCE",
        "UNPACK_EX",

        "BUILD_SLICE",

        # Functions / generators
        "MAKE_FUNCTION",
        "YIELD_VALUE",
        "SEND",
        "GET_YIELD_FROM_ITER",
        "YIELD_FROM",

        # Returns / exceptions
        "RETURN_VALUE",
        "RETURN_CONST",

        "RAISE_VARARGS",
        "RERAISE",

        "PUSH_EXC_INFO",
        "POP_EXCEPT",
        "CHECK_EXC_MATCH",
        "WITH_EXCEPT_START",

        # Context managers
        "BEFORE_WITH",
        "BEFORE_ASYNC_WITH",

        # Async
        "GET_AWAITABLE",
        "GET_AITER",
        "GET_ANEXT",
        "END_ASYNC_FOR",
    }
)


# ---------------------------------------------------------------------------
# Operator normalization
# ---------------------------------------------------------------------------

_BINARY_OPERATORS = {
    "+": "+",
    "-": "-",
    "*": "*",
    "@": "@",
    "/": "/",
    "//": "//",
    "%": "%",
    "**": "**",
    "<<": "<<",
    ">>": ">>",
    "&": "&",
    "|": "|",
    "^": "^",

    "+=": "+=",
    "-=": "-=",
    "*=": "*=",
    "@=": "@=",
    "/=": "/=",
    "//=": "//=",
    "%=": "%=",
    "**=": "**=",
    "<<=": "<<=",
    ">>=": ">>=",
    "&=": "&=",
    "|=": "|=",
    "^=": "^=",
}


_BINARY_OPERATOR_NAMES = {
    "NB_ADD": "+",
    "NB_SUBTRACT": "-",
    "NB_MULTIPLY": "*",
    "NB_MATRIX_MULTIPLY": "@",
    "NB_TRUE_DIVIDE": "/",
    "NB_FLOOR_DIVIDE": "//",
    "NB_REMAINDER": "%",
    "NB_POWER": "**",
    "NB_LSHIFT": "<<",
    "NB_RSHIFT": ">>",
    "NB_AND": "&",
    "NB_OR": "|",
    "NB_XOR": "^",
}


def normalize_binary_operator(
    argrepr: str,
) -> str:
    """
    Normalize the representation of BINARY_OP.

    CPython versions can expose different representations through
    dis.Instruction.argrepr. The decompiler only needs the semantic
    operator.
    """

    if argrepr in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[argrepr]

    if argrepr in _BINARY_OPERATOR_NAMES:
        return _BINARY_OPERATOR_NAMES[argrepr]

    return argrepr


# ---------------------------------------------------------------------------
# Instruction normalization
# ---------------------------------------------------------------------------

def _get_jump_target(
    instruction: dis.Instruction,
) -> int | None:
    """
    Return a normalized jump target.

    CPython exposes jump targets through argval for modern Python
    versions. Keeping this logic here means the rest of PyArch does
    not need to care about CPython's representation.
    """

    if instruction.opcode in dis.hasjabs:
        if isinstance(instruction.argval, int):
            return instruction.argval

    if instruction.opcode in dis.hasjrel:
        if isinstance(instruction.argval, int):
            return instruction.argval

    if "JUMP" in instruction.opname:
        if isinstance(instruction.argval, int):
            return instruction.argval

    if instruction.opname == "FOR_ITER":
        if isinstance(instruction.argval, int):
            return instruction.argval

    if instruction.opname == "SEND":
        if isinstance(instruction.argval, int):
            return instruction.argval

    return None


def _normalize_instruction(
    instruction: dis.Instruction,
) -> TInstruction | None:
    """
    Convert one CPython instruction to a semantic instruction.

    Implementation-only instructions such as CACHE and EXTENDED_ARG
    disappear here because they have no source-level meaning.
    """

    original = instruction.opname

    # These instructions exist for CPython's internal execution/
    # specialization machinery and should never reach the source
    # reconstruction layer.
    if original in {
        "CACHE",
        "EXTENDED_ARG",
    }:
        return None

    op = original
    arg = instruction.arg
    argval = instruction.argval
    argrepr = instruction.argrepr

    # ---------------------------------------------------------------
    # BINARY_OP
    # ---------------------------------------------------------------

    if original == "BINARY_OP":
        argval = normalize_binary_operator(
            instruction.argrepr
        )

    # ---------------------------------------------------------------
    # Older / alternative call instructions
    # ---------------------------------------------------------------

    # These aliases allow future Python-version adapters to map old
    # call instructions to the same semantic CALL operation.
    if original in {
        "CALL_FUNCTION",
        "CALL_METHOD",
    }:
        op = "CALL"

    # ---------------------------------------------------------------
    # Older jump spelling
    # ---------------------------------------------------------------

    jump_aliases = {
        "JUMP_ABSOLUTE": "JUMP_FORWARD",
        "POP_JUMP_IF_FALSE": "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_IF_TRUE": "POP_JUMP_FORWARD_IF_TRUE",
    }

    op = jump_aliases.get(
        op,
        op,
    )

    target = _get_jump_target(
        instruction
    )

    return TInstruction(
        offset=instruction.offset,
        op=op,
        arg=arg,
        argval=argval,
        argrepr=argrepr,
        target=target,
        original=original,
    )


# ---------------------------------------------------------------------------
# Public translation API
# ---------------------------------------------------------------------------

def translate_instructions(
    instructions: list[dis.Instruction],
) -> list[TInstruction]:
    """
    Translate a list of CPython instructions.

    Instructions that have no semantic meaning are removed.
    """

    translated: list[TInstruction] = []

    for instruction in instructions:
        result = _normalize_instruction(
            instruction
        )

        if result is None:
            continue

        translated.append(result)

    return translated


def translate_code(
    code: CodeType,
) -> list[TInstruction]:
    """
    Translate a Python CodeType into normalized PyArch instructions.
    """

    instructions = list(
        dis.get_instructions(code)
    )

    return translate_instructions(
        instructions
    )


# ---------------------------------------------------------------------------
# Version information
# ---------------------------------------------------------------------------

def running_python_version() -> tuple[int, int]:
    """Return the running Python major/minor version."""

    return (
        sys.version_info.major,
        sys.version_info.minor,
    )


def running_python_version_string() -> str:
    """Return the running Python version as 'major.minor'."""

    major, minor = running_python_version()

    return f"{major}.{minor}"


def is_supported_semantic_op(
    op: str,
) -> bool:
    """
    Return whether the semantic operation belongs to PyArch's
    normalized instruction vocabulary.
    """

    return op in SEMANTIC_OPS
