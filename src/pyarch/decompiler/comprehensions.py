from __future__ import annotations

"""
Comprehension reconstruction for PyArch.

Python bytecode has changed significantly across versions.

Older versions commonly compile comprehensions into nested code
objects such as:

    <listcomp>
    <setcomp>
    <dictcomp>
    <genexpr>

Newer CPython versions, including Python 3.13, can compile
comprehensions directly into the surrounding code object.

This module therefore supports both representations.

The module is intentionally conservative. It detects and describes
comprehension structures, while the main decompiler engine is
responsible for reconstructing their final expressions and AST.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import CodeType
from typing import Iterable

from .ir import IRExpression


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ComprehensionError(Exception):
    """Raised when comprehension reconstruction fails."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ComprehensionType(str, Enum):
    """Supported Python comprehension types."""

    LIST = "list"
    SET = "set"
    DICT = "dict"
    GENERATOR = "generator"


_COMPREHENSION_NAMES: dict[str, ComprehensionType] = {
    "<listcomp>": ComprehensionType.LIST,
    "<setcomp>": ComprehensionType.SET,
    "<dictcomp>": ComprehensionType.DICT,
    "<genexpr>": ComprehensionType.GENERATOR,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComprehensionClause:
    """
    One ``for`` clause.

    Example:

        [x * 2 for x in values if x > 0]

    is represented conceptually as:

        target   = x
        iterable = values
        conditions = [x > 0]
    """

    target: IRExpression | None = None
    iterable: IRExpression | None = None
    conditions: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class Comprehension:
    """Initial representation of a Python comprehension."""

    kind: ComprehensionType

    element: IRExpression | None = None

    key: IRExpression | None = None
    value: IRExpression | None = None

    clauses: list[ComprehensionClause] = field(
        default_factory=list
    )

    source: CodeType | None = None

    # True when the comprehension is detected directly in the
    # surrounding bytecode rather than through a nested <listcomp>
    # style code object.
    inline: bool = False


@dataclass(slots=True)
class ComprehensionStructure:
    """
    Low-level structural information about a comprehension.

    This is deliberately independent from the final Python AST.
    """

    kind: ComprehensionType

    has_for: bool = False
    has_if: bool = False
    multiple_for: bool = False
    is_async: bool = False

    # True when the comprehension is compiled directly into the
    # surrounding code object, as can happen on Python 3.13+.
    inline: bool = False

    append_opcode: str | None = None

    instruction_count: int = 0

    names: list[str] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Opcode groups
# ---------------------------------------------------------------------------


_FOR_ITER = {
    "FOR_ITER",
}

_ITERATION_OPS = {
    "GET_ITER",
    "GET_AITER",
    "GET_ANEXT",
    "FOR_ITER",
}

_FILTER_JUMPS = {
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "JUMP_IF_FALSE_OR_POP",
}

_LIST_APPEND = {
    "LIST_APPEND",
}

_SET_APPEND = {
    "SET_ADD",
}

_DICT_APPEND = {
    "MAP_ADD",
}

_YIELD = {
    "YIELD_VALUE",
    "YIELD_FROM",
}

_STORE_TARGETS = {
    "STORE_FAST",
    "STORE_NAME",
    "STORE_GLOBAL",
    "STORE_DEREF",
    "STORE_FAST_LOAD_FAST",
}


# ---------------------------------------------------------------------------
# Instruction helpers
# ---------------------------------------------------------------------------


def _get_instructions(
    code: CodeType,
):
    """
    Return CPython's decoded instructions.

    Keeping this isolated makes it easier to replace this with
    PyArch's normalized TInstruction stream later.
    """

    import dis

    return list(
        dis.get_instructions(code)
    )


def _instruction_names(
    code: CodeType,
) -> list[str]:
    return [
        instruction.opname
        for instruction in _get_instructions(code)
    ]


def _contains_any(
    names: Iterable[str],
    candidates: set[str],
) -> bool:
    return any(
        name in candidates
        for name in names
    )


# ---------------------------------------------------------------------------
# Code-object classification
# ---------------------------------------------------------------------------


def is_comprehension_code(
    code: CodeType,
) -> bool:
    """
    Return whether ``code`` is a traditional nested comprehension
    code object.
    """

    return code.co_name in _COMPREHENSION_NAMES


def comprehension_type(
    code: CodeType,
) -> ComprehensionType:
    """Return the type of a nested comprehension code object."""

    try:
        return _COMPREHENSION_NAMES[
            code.co_name
        ]
    except KeyError as error:
        raise ComprehensionError(
            f"Not a comprehension code object: "
            f"{code.co_name!r}"
        ) from error


# ---------------------------------------------------------------------------
# Nested code objects
# ---------------------------------------------------------------------------


def nested_code_objects(
    code: CodeType,
) -> list[CodeType]:
    """Return code objects directly contained in ``code``."""

    return [
        constant
        for constant in code.co_consts
        if isinstance(
            constant,
            CodeType,
        )
    ]


def iter_nested_code_objects(
    code: CodeType,
):
    """Recursively yield nested code objects."""

    for nested in nested_code_objects(code):
        yield nested
        yield from iter_nested_code_objects(
            nested
        )


def discover_comprehensions(
    code: CodeType,
) -> list[CodeType]:
    """
    Recursively find traditional comprehension code objects.

    This works for Python versions where comprehensions are emitted
    as nested ``<listcomp>`` / ``<setcomp>`` / ``<dictcomp>`` /
    ``<genexpr>`` objects.
    """

    return [
        nested
        for nested in iter_nested_code_objects(code)
        if is_comprehension_code(nested)
    ]


# ---------------------------------------------------------------------------
# Comprehension markers
# ---------------------------------------------------------------------------


def _detect_append_opcode(
    names: list[str],
) -> str | None:
    """Determine which container receives the comprehension result."""

    if _contains_any(
        names,
        _LIST_APPEND,
    ):
        return "LIST_APPEND"

    if _contains_any(
        names,
        _SET_APPEND,
    ):
        return "SET_ADD"

    if _contains_any(
        names,
        _DICT_APPEND,
    ):
        return "MAP_ADD"

    if _contains_any(
        names,
        _YIELD,
    ):
        return "YIELD_VALUE"

    return None


def _type_from_append_opcode(
    opcode: str | None,
) -> ComprehensionType | None:
    """Map a container-building opcode to a comprehension type."""

    if opcode == "LIST_APPEND":
        return ComprehensionType.LIST

    if opcode == "SET_ADD":
        return ComprehensionType.SET

    if opcode == "MAP_ADD":
        return ComprehensionType.DICT

    if opcode == "YIELD_VALUE":
        return ComprehensionType.GENERATOR

    return None


# ---------------------------------------------------------------------------
# Inline comprehension detection
# ---------------------------------------------------------------------------


def detect_inline_comprehension(
    code: CodeType,
) -> ComprehensionStructure | None:
    """
    Detect a comprehension compiled directly into ``code``.

    Python 3.13 can emit the comprehension loop directly instead of
    creating a ``<listcomp>`` code object.

    Typical structural markers are:

        BUILD_LIST
        GET_ITER
        FOR_ITER
        ...
        LIST_APPEND

    The exact bytecode sequence is intentionally not hard-coded,
    because it differs between Python versions.
    """

    instructions = _get_instructions(
        code
    )

    names = [
        instruction.opname
        for instruction in instructions
    ]

    append_opcode = _detect_append_opcode(
        names
    )

    if append_opcode is None:
        return None

    for_count = names.count(
        "FOR_ITER"
    )

    # A container-building opcode alone is not enough.
    # We need an iteration structure as well.
    if for_count == 0:
        return None

    kind = _type_from_append_opcode(
        append_opcode
    )

    if kind is None:
        return None

    return ComprehensionStructure(
        kind=kind,
        has_for=True,
        has_if=_contains_any(
            names,
            _FILTER_JUMPS,
        ),
        multiple_for=for_count > 1,
        is_async=_contains_any(
            names,
            {"GET_AITER", "GET_ANEXT"},
        ),
        append_opcode=append_opcode,
        instruction_count=len(
            instructions
        ),
        names=sorted(
            {
                name
                for name in code.co_names
                if isinstance(name, str)
            }
        ),
    )


# ---------------------------------------------------------------------------
# Traditional comprehension analysis
# ---------------------------------------------------------------------------


def comprehension_flags(
    code: CodeType,
) -> dict[str, bool]:
    """Return useful flags for a nested comprehension."""

    if not is_comprehension_code(code):
        raise ComprehensionError(
            f"{code.co_name!r} is not a comprehension."
        )

    names = _instruction_names(
        code
    )

    return {
        "is_generator": (
            comprehension_type(code)
            == ComprehensionType.GENERATOR
        ),
        "is_async": _contains_any(
            names,
            {"GET_AITER", "GET_ANEXT"},
        ),
    }


def comprehension_element_names(
    code: CodeType,
) -> list[str]:
    """
    Return names referenced by a comprehension.

    This is metadata only; it does not claim that every name is part
    of the resulting expression.
    """

    if not is_comprehension_code(code):
        raise ComprehensionError(
            f"{code.co_name!r} is not a comprehension."
        )

    return sorted(
        {
            name
            for name in code.co_names
            if isinstance(name, str)
        }
    )


def has_for_iteration(
    code: CodeType,
) -> bool:
    """Return whether the bytecode contains a FOR_ITER."""

    return _contains_any(
        _instruction_names(code),
        _FOR_ITER,
    )


def has_filters(
    code: CodeType,
) -> bool:
    """Return whether the bytecode appears to contain an ``if``."""

    return _contains_any(
        _instruction_names(code),
        _FILTER_JUMPS,
    )


def has_multiple_iterations(
    code: CodeType,
) -> bool:
    """Return whether multiple comprehension loops are present."""

    return (
        _instruction_names(code).count(
            "FOR_ITER"
        )
        > 1
    )


# ---------------------------------------------------------------------------
# Unified analysis
# ---------------------------------------------------------------------------


def analyze_comprehension(
    code: CodeType,
) -> Comprehension:
    """
    Create a comprehension representation.

    Both nested and inline representations are accepted.
    """

    if is_comprehension_code(code):
        return Comprehension(
            kind=comprehension_type(code),
            source=code,
            inline=False,
        )

    inline = detect_inline_comprehension(
        code
    )

    if inline is None:
        raise ComprehensionError(
            f"Could not detect a comprehension "
            f"in {code.co_name!r}."
        )

    return Comprehension(
        kind=inline.kind,
        source=code,
        inline=True,
    )


def analyze_comprehension_structure(
    code: CodeType,
) -> ComprehensionStructure:
    """
    Analyze either a traditional nested comprehension or an inline
    comprehension.
    """

    if is_comprehension_code(code):
        names = _instruction_names(
            code
        )

        return ComprehensionStructure(
            kind=kind,
            has_for=True,
            has_if=_contains_any(
                names,
                _FILTER_JUMPS,
            ),
            multiple_for=for_count > 1,
            is_async=_contains_any(
                names,
                {"GET_AITER", "GET_ANEXT"},
            ),
            inline=True,
            append_opcode=append_opcode,
            instruction_count=len(instructions),
            names=sorted(
                {
                    name
                    for name in code.co_names
                    if isinstance(name, str)
                }
            ),
        )

    inline = detect_inline_comprehension(
        code
    )

    if inline is None:
        raise ComprehensionError(
            f"No comprehension detected in "
            f"{code.co_name!r}."
        )

    return inline


# ---------------------------------------------------------------------------
# Discovery in arbitrary code
# ---------------------------------------------------------------------------


def contains_comprehension(
    code: CodeType,
) -> bool:
    """Return whether a code object contains a comprehension."""

    if is_comprehension_code(code):
        return True

    if detect_inline_comprehension(code) is not None:
        return True

    return any(
        contains_comprehension(nested)
        for nested in nested_code_objects(code)
    )


def find_comprehensions(
    code: CodeType,
) -> list[Comprehension]:
    """
    Find all detectable comprehensions below ``code``.

    Both nested and inline forms are considered.
    """

    result: list[Comprehension] = []

    def walk(
        current: CodeType,
        *,
        root: bool = False,
    ) -> None:
        if not root and is_comprehension_code(
            current
        ):
            result.append(
                analyze_comprehension(
                    current
                )
            )
            return

        inline = detect_inline_comprehension(
            current
        )

        if inline is not None:
            result.append(
                Comprehension(
                    kind=inline.kind,
                    source=current,
                    inline=True,
                )
            )

        for nested in nested_code_objects(
            current
        ):
            walk(
                nested,
                root=False,
            )

    walk(
        code,
        root=True,
    )

    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_expression(
    expression: IRExpression,
) -> str:
    from .expressions import render_expression

    return render_expression(
        expression
    ).text


def _render_clause(
    clause: ComprehensionClause,
) -> str:
    if (
        clause.target is None
        or clause.iterable is None
    ):
        raise ComprehensionError(
            "Cannot render an incomplete "
            "comprehension clause."
        )

    result = (
        "for "
        + _render_expression(
            clause.target
        )
        + " in "
        + _render_expression(
            clause.iterable
        )
    )

    for condition in clause.conditions:
        result += (
            " if "
            + _render_expression(
                condition
            )
        )

    return result


def render_comprehension(
    comprehension: Comprehension,
) -> str:
    """
    Render a fully reconstructed comprehension.

    This is a temporary/debug renderer. The final decompiler will
    produce Python AST nodes and use ``ast.unparse``.
    """

    clauses = " ".join(
        _render_clause(clause)
        for clause in comprehension.clauses
    )

    suffix = (
        " " + clauses
        if clauses
        else ""
    )

    if comprehension.kind == (
        ComprehensionType.DICT
    ):
        if (
            comprehension.key is None
            or comprehension.value is None
        ):
            raise ComprehensionError(
                "Dictionary comprehension is incomplete."
            )

        expression = (
            _render_expression(
                comprehension.key
            )
            + ": "
            + _render_expression(
                comprehension.value
            )
        )

        return "{" + expression + suffix + "}"

    if comprehension.element is None:
        raise ComprehensionError(
            "Comprehension element is missing."
        )

    element = _render_expression(
        comprehension.element
    )

    if comprehension.kind == (
        ComprehensionType.LIST
    ):
        return "[" + element + suffix + "]"

    if comprehension.kind == (
        ComprehensionType.SET
    ):
        return "{" + element + suffix + "}"

    if comprehension.kind == (
        ComprehensionType.GENERATOR
    ):
        return "(" + element + suffix + ")"

    raise ComprehensionError(
        f"Unsupported comprehension type: "
        f"{comprehension.kind}"
    )


# ---------------------------------------------------------------------------
# Debugging
# ---------------------------------------------------------------------------


def describe_comprehension(
    code: CodeType,
) -> str:
    """Return a compact human-readable description."""

    structure = analyze_comprehension_structure(
        code
    )

    flags: list[str] = []

    if structure.has_if:
        flags.append("if")

    if structure.multiple_for:
        flags.append("multiple-for")

    if structure.is_async:
        flags.append("async")

    if structure.inline:
        flags.append("inline")

    suffix = (
        ", " + ", ".join(flags)
        if flags
        else ""
    )

    return (
        f"{structure.kind.value} "
        f"comprehension"
        f"{suffix}"
    )
