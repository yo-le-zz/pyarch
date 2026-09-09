from __future__ import annotations

"""
Python expression reconstruction.

This module converts PyArch IR expressions into valid Python source.

It does not handle statements or control flow.
"""


from dataclasses import dataclass
from typing import Any

from .ir import (
    Attribute,
    BinaryOp,
    Call,
    Compare,
    Constant,
    DictExpr,
    IRExpression,
    ListExpr,
    Name,
    SetExpr,
    SliceExpr,
    Subscript,
    TupleExpr,
    UnaryOp,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExpressionError(Exception):
    """Raised when an IR expression cannot be rendered."""


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


# Higher number = binds more strongly.
_PRECEDENCE = {
    "lambda": 5,

    "or": 10,
    "and": 20,

    "not": 30,

    "in": 40,
    "not in": 40,
    "is": 40,
    "is not": 40,
    "<": 40,
    "<=": 40,
    "==": 40,
    "!=": 40,
    ">": 40,
    ">=": 40,

    "|": 50,
    "^": 60,
    "&": 70,

    "<<": 80,
    ">>": 80,

    "+": 90,
    "-": 90,

    "*": 100,
    "@": 100,
    "/": 100,
    "//": 100,
    "%": 100,

    "**": 110,

    "unary": 120,

    "call": 130,
    "attribute": 130,
    "subscript": 130,

    "atom": 140,
}


_BINARY_PRECEDENCE = {
    "|": 50,
    "^": 60,
    "&": 70,
    "<<": 80,
    ">>": 80,
    "+": 90,
    "-": 90,
    "*": 100,
    "@": 100,
    "/": 100,
    "//": 100,
    "%": 100,
    "**": 110,
}


# ---------------------------------------------------------------------------
# Render result
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RenderedExpression:
    """Rendered Python expression with its precedence."""

    source: str
    precedence: int


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def render_expression(
    expression: IRExpression,
) -> str:
    """Render an IR expression as valid Python source."""

    return _render(
        expression
    ).source


def _render(
    expression: IRExpression,
) -> RenderedExpression:

    # ------------------------------------------------------------------
    # Constant
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Constant,
    ):
        return RenderedExpression(
            source=repr(expression.value),
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Name,
    ):
        if not expression.name:
            raise ExpressionError(
                "Cannot render an empty name."
            )

        return RenderedExpression(
            source=expression.name,
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Attribute
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Attribute,
    ):
        value = _render(
            expression.value
        )

        value_source = _parenthesize(
            value,
            _PRECEDENCE["attribute"],
        )

        return RenderedExpression(
            source=(
                f"{value_source}."
                f"{expression.name}"
            ),
            precedence=_PRECEDENCE[
                "attribute"
            ],
        )

    # ------------------------------------------------------------------
    # Subscript
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Subscript,
    ):
        value = _render(
            expression.value
        )

        index = _render(
            expression.index
        )

        value_source = _parenthesize(
            value,
            _PRECEDENCE["subscript"],
        )

        return RenderedExpression(
            source=(
                f"{value_source}"
                f"[{index.source}]"
            ),
            precedence=_PRECEDENCE[
                "subscript"
            ],
        )

    # ------------------------------------------------------------------
    # Unary
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        UnaryOp,
    ):
        operand = _render(
            expression.operand
        )

        operand_source = _parenthesize(
            operand,
            _PRECEDENCE["unary"],
        )

        if expression.op == "not":
            source = f"not {operand_source}"

        elif expression.op in {
            "-",
            "+",
            "~",
        }:
            source = (
                f"{expression.op}"
                f"{operand_source}"
            )

        else:
            raise ExpressionError(
                f"Unsupported unary operator: "
                f"{expression.op!r}"
            )

        return RenderedExpression(
            source=source,
            precedence=_PRECEDENCE[
                "unary"
            ],
        )

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        BinaryOp,
    ):
        operator = expression.op

        if operator not in _BINARY_PRECEDENCE:
            raise ExpressionError(
                f"Unsupported binary operator: "
                f"{operator!r}"
            )

        precedence = _BINARY_PRECEDENCE[
            operator
        ]

        left = _render(
            expression.left
        )

        right = _render(
            expression.right
        )

        left_source = _parenthesize(
            left,
            precedence,
        )

        # Exponentiation is right-associative.
        if operator == "**":
            right_source = _parenthesize(
                right,
                precedence - 1,
            )
        else:
            right_source = _parenthesize(
                right,
                precedence + 1,
            )

        return RenderedExpression(
            source=(
                f"{left_source} "
                f"{operator} "
                f"{right_source}"
            ),
            precedence=precedence,
        )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Compare,
    ):
        operator = expression.op

        if operator not in {
            "<",
            "<=",
            "==",
            "!=",
            ">",
            ">=",
            "in",
            "not in",
            "is",
            "is not",
        }:
            raise ExpressionError(
                f"Unsupported comparison operator: "
                f"{operator!r}"
            )

        left = _render(
            expression.left
        )

        right = _render(
            expression.right
        )

        precedence = _PRECEDENCE[
            operator
        ]

        return RenderedExpression(
            source=(
                f"{_parenthesize(left, precedence)} "
                f"{operator} "
                f"{_parenthesize(right, precedence + 1)}"
            ),
            precedence=precedence,
        )

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        Call,
    ):
        function = _render(
            expression.function
        )

        function_source = _parenthesize(
            function,
            _PRECEDENCE["call"],
        )

        arguments = [
            _render(argument).source
            for argument in expression.args
        ]

        for name, value in expression.keywords:
            rendered = _render(value).source

            if name is None:
                arguments.append(
                    f"**{rendered}"
                )
            else:
                arguments.append(
                    f"{name}={rendered}"
                )

        return RenderedExpression(
            source=(
                f"{function_source}"
                f"({', '.join(arguments)})"
            ),
            precedence=_PRECEDENCE[
                "call"
            ],
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        ListExpr,
    ):
        elements = [
            _render(element).source
            for element in expression.elements
        ]

        return RenderedExpression(
            source=(
                "["
                + ", ".join(elements)
                + "]"
            ),
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Tuple
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        TupleExpr,
    ):
        elements = [
            _render(element).source
            for element in expression.elements
        ]

        if len(elements) == 0:
            source = "()"

        elif len(elements) == 1:
            source = (
                f"({elements[0]},)"
            )

        else:
            source = (
                "("
                + ", ".join(elements)
                + ")"
            )

        return RenderedExpression(
            source=source,
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Set
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        SetExpr,
    ):
        elements = [
            _render(element).source
            for element in expression.elements
        ]

        if not elements:
            # An empty set cannot be represented by {}.
            source = "set()"
        else:
            source = (
                "{"
                + ", ".join(elements)
                + "}"
            )

        return RenderedExpression(
            source=source,
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        DictExpr,
    ):
        entries: list[str] = []

        for key, value in expression.entries:
            value_source = _render(
                value
            ).source

            if key is None:
                entries.append(
                    f"**{value_source}"
                )
                continue

            key_source = _render(
                key
            ).source

            entries.append(
                f"{key_source}: "
                f"{value_source}"
            )

        return RenderedExpression(
            source=(
                "{"
                + ", ".join(entries)
                + "}"
            ),
            precedence=_PRECEDENCE["atom"],
        )

    # ------------------------------------------------------------------
    # Slice
    # ------------------------------------------------------------------

    if isinstance(
        expression,
        SliceExpr,
    ):
        lower = (
            _render(
                expression.lower
            ).source
            if expression.lower is not None
            else ""
        )

        upper = (
            _render(
                expression.upper
            ).source
            if expression.upper is not None
            else ""
        )

        if expression.step is None:
            source = (
                f"{lower}:{upper}"
            )
        else:
            step = _render(
                expression.step
            ).source

            source = (
                f"{lower}:{upper}:{step}"
            )

        return RenderedExpression(
            source=source,
            precedence=_PRECEDENCE[
                "atom"
            ],
        )

    raise ExpressionError(
        "Unsupported IR expression: "
        f"{type(expression).__name__}"
    )


# ---------------------------------------------------------------------------
# Parentheses
# ---------------------------------------------------------------------------


def _parenthesize(
    expression: RenderedExpression,
    minimum_precedence: int,
) -> str:
    """
    Add parentheses when an expression binds too weakly.

    Equal precedence normally does not require parentheses because
    the caller handles associativity where necessary.
    """

    if (
        expression.precedence
        < minimum_precedence
    ):
        return (
            f"({expression.source})"
        )

    return expression.source


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def render_constant(
    value: Any,
) -> str:
    """Render a raw Python constant."""

    return render_expression(
        Constant(value=value)
    )


def render_name(
    name: str,
) -> str:
    """Render a Python name."""

    return render_expression(
        Name(name=name)
    )


def render_binary(
    operator: str,
    left: IRExpression,
    right: IRExpression,
) -> str:
    """Render a binary expression."""

    return render_expression(
        BinaryOp(
            op=operator,
            left=left,
            right=right,
        )
    )


def render_call(
    function: IRExpression,
    args: list[IRExpression] | None = None,
) -> str:
    """Render a function call."""

    return render_expression(
        Call(
            function=function,
            args=[] if args is None else args,
        )
    )