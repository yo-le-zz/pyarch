from __future__ import annotations

"""
Comprehension reconstruction for PyArch.

Python compiles comprehensions into separate nested code objects.
This module identifies those objects and extracts the information
needed by the decompiler engine.

Supported forms:

    [expr for x in iterable]
    {expr for x in iterable}
    {key: value for x in iterable}
    (expr for x in iterable)

Nested loops and ``if`` clauses are represented structurally and are
handled by later reconstruction passes.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import CodeType

from .ir import IRExpression, IRStatement


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ComprehensionError(Exception):
    """Raised when comprehension reconstruction fails."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ComprehensionType(str, Enum):
    """Python comprehension kinds."""

    LIST = "list"
    SET = "set"
    DICT = "dict"
    GENERATOR = "generator"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComprehensionClause:
    """
    One ``for`` clause inside a comprehension.

    Example:

        [x for x in values if x > 0]

    produces one clause whose target is ``x`` and whose iterable is
    ``values``.
    """

    target: IRExpression | None = None

    iterable: IRExpression | None = None

    conditions: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class Comprehension:
    """Reconstructed comprehension."""

    kind: ComprehensionType

    element: IRExpression | None = None

    key: IRExpression | None = None

    value: IRExpression | None = None

    clauses: list[ComprehensionClause] = field(
        default_factory=list
    )

    source: CodeType | None = None


# ---------------------------------------------------------------------------
# Code-object classification
# ---------------------------------------------------------------------------


_COMPREHENSION_NAMES = {
    "<listcomp>": ComprehensionType.LIST,
    "<setcomp>": ComprehensionType.SET,
    "<dictcomp>": ComprehensionType.DICT,
    "<genexpr>": ComprehensionType.GENERATOR,
}


def is_comprehension_code(
    code: CodeType,
) -> bool:
    """Return whether a code object represents a comprehension."""

    return code.co_name in _COMPREHENSION_NAMES


def comprehension_type(
    code: CodeType,
) -> ComprehensionType:
    """
    Return the comprehension type encoded by a code object.
    """

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
    """Return code objects directly contained in a code object."""

    return [
        constant
        for constant in code.co_consts
        if isinstance(
            constant,
            CodeType,
        )
    ]


def discover_comprehensions(
    code: CodeType,
) -> list[CodeType]:
    """
    Find comprehension code objects directly contained in ``code``.
    """

    return [
        nested
        for nested in nested_code_objects(code)
        if is_comprehension_code(nested)
    ]


# ---------------------------------------------------------------------------
# Code-object metadata
# ---------------------------------------------------------------------------


def comprehension_flags(
    code: CodeType,
) -> dict[str, bool]:
    """
    Return useful metadata about a comprehension code object.
    """

    if not is_comprehension_code(code):
        raise ComprehensionError(
            f"{code.co_name!r} is not a comprehension."
        )

    return {
        "is_generator": (
            comprehension_type(code)
            == ComprehensionType.GENERATOR
        ),
        "is_async": bool(
            code.co_flags & 0x0200
        ),
    }


def comprehension_element_names(
    code: CodeType,
) -> list[str]:
    """
    Return names referenced by a comprehension.

    This is useful for debugging and later stack reconstruction.
    It is not yet a semantic representation of the expression.
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


# ---------------------------------------------------------------------------
# Bytecode markers
# ---------------------------------------------------------------------------


_FOR_ITER = {
    "FOR_ITER",
}

_STORE_TARGETS = {
    "STORE_FAST",
    "STORE_NAME",
    "STORE_GLOBAL",
    "STORE_DEREF",
}

_FILTER_JUMPS = {
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "JUMP_IF_FALSE_OR_POP",
}


def _instruction_names(
    code: CodeType,
) -> list[str]:
    """Return normalized opcode names from a code object."""

    import dis

    return [
        instruction.opname
        for instruction in dis.get_instructions(
            code
        )
    ]


def has_for_iteration(
    code: CodeType,
) -> bool:
    """Return whether the comprehension contains FOR_ITER."""

    return any(
        opname in _FOR_ITER
        for opname in _instruction_names(code)
    )


def has_filters(
    code: CodeType,
) -> bool:
    """
    Return whether the comprehension appears to contain ``if``
    filtering.
    """

    return any(
        opname in _FILTER_JUMPS
        for opname in _instruction_names(code)
    )


def has_multiple_iterations(
    code: CodeType,
) -> bool:
    """Return whether multiple FOR_ITER instructions exist."""

    return (
        _instruction_names(code).count(
            "FOR_ITER"
        )
        > 1
    )


# ---------------------------------------------------------------------------
# Structural analysis
# ---------------------------------------------------------------------------


def analyze_comprehension(
    code: CodeType,
) -> Comprehension:
    """
    Create an initial comprehension representation.

    Expression and iterator values are intentionally left unresolved.
    The stack engine will recover those values later.
    """

    kind = comprehension_type(
        code
    )

    return Comprehension(
        kind=kind,
        source=code,
    )


def analyze_comprehension_structure(
    code: CodeType,
) -> dict[str, object]:
    """
    Return structural information useful to the decompiler engine.
    """

    if not is_comprehension_code(code):
        raise ComprehensionError(
            f"{code.co_name!r} is not a comprehension."
        )

    return {
        "kind": comprehension_type(code).value,
        "has_for": has_for_iteration(code),
        "has_if": has_filters(code),
        "multiple_for": has_multiple_iterations(code),
        "names": comprehension_element_names(code),
        **comprehension_flags(code),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_expression(
    expression: IRExpression,
) -> str:
    """Render an IR expression."""

    from .expressions import render_expression

    return render_expression(
        expression
    ).text


def _render_clause(
    clause: ComprehensionClause,
) -> str:
    """Render one comprehension clause."""

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
    Render a fully populated comprehension.

    This is primarily a debugging renderer. The final writer will
    construct Python AST nodes.
    """

    clauses = " ".join(
        _render_clause(clause)
        for clause in comprehension.clauses
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

        return (
            "{"
            + expression
            + (" " + clauses if clauses else "")
            + "}"
        )

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
        return (
            "["
            + element
            + (" " + clauses if clauses else "")
            + "]"
        )

    if comprehension.kind == (
        ComprehensionType.SET
    ):
        return (
            "{"
            + element
            + (" " + clauses if clauses else "")
            + "}"
        )

    if comprehension.kind == (
        ComprehensionType.GENERATOR
    ):
        return (
            "("
            + element
            + (" " + clauses if clauses else "")
            + ")"
        )

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
    """Return a compact description of a comprehension."""

    structure = analyze_comprehension_structure(
        code
    )

    flags: list[str] = []

    if structure["has_if"]:
        flags.append("if")

    if structure["multiple_for"]:
        flags.append("multiple-for")

    if structure["is_async"]:
        flags.append("async")

    suffix = (
        ", " + ", ".join(flags)
        if flags
        else ""
    )

    return (
        f"{structure['kind']} "
        f"comprehension"
        f"{suffix}"
    )