from __future__ import annotations

"""
Python statement reconstruction.

This module converts semantic bytecode operations and the virtual
evaluation stack into PyArch IR statements.

It intentionally does not try to reconstruct structured control flow
(if/while/for/try). That belongs to later passes.
"""


from dataclasses import dataclass, field

from .expressions import render_expression
from .ir import (
    Assign,
    Delete,
    ExpressionStatement,
    IRExpression,
    IRStatement,
    Name,
    Pass,
    Return,
)
from .stack import (
    StackError,
    StackValue,
    VirtualStack,
)
from .translate import TInstruction


class StatementError(Exception):
    """Raised when statement reconstruction fails."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StatementResult:
    """
    Result of processing a sequence of instructions.

    `statements` contains reconstructed source-level statements.

    `stack` is the remaining virtual stack state.
    """

    statements: list[IRStatement] = field(
        default_factory=list
    )

    stack: VirtualStack = field(
        default_factory=VirtualStack
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_name(
    instruction: TInstruction,
) -> str:
    """Extract a valid name from a STORE/DELETE instruction."""

    if isinstance(
        instruction.argval,
        str,
    ):
        return instruction.argval

    if isinstance(
        instruction.argrepr,
        str,
    ):
        value = instruction.argrepr.strip()

        if value:
            return value

    raise StatementError(
        f"{instruction.op} has no valid name."
    )


def _pop_expression(
    stack: VirtualStack,
) -> IRExpression:
    """Pop an expression with a statement-level error."""

    try:
        return stack.pop_expression()
    except StackError as error:
        raise StatementError(
            "Expected an expression on the evaluation stack."
        ) from error


def _consume_return_const(
    instruction: TInstruction,
) -> IRExpression:
    """
    RETURN_CONST directly contains the returned constant on modern
    CPython versions.
    """

    return_value = instruction.argval

    # CPython's dis representation may expose None through argval while
    # argrepr contains the actual representation. For RETURN_CONST,
    # None is also a perfectly valid return value, so preserve it.
    from .ir import Constant

    return Constant(
        value=return_value
    )


# ---------------------------------------------------------------------------
# Individual instructions
# ---------------------------------------------------------------------------


def process_instruction(
    stack: VirtualStack,
    instruction: TInstruction,
) -> IRStatement | None:
    """
    Process one instruction.

    Returns a statement when the instruction produces one.
    """

    op = instruction.op

    # ------------------------------------------------------------------
    # Stack-only instructions
    # ------------------------------------------------------------------

    if op in {
        "RESUME",
        "NOP",
        "PRECALL",
        "KW_NAMES",
    }:
        return None

    # ------------------------------------------------------------------
    # Constants / names / expressions
    # ------------------------------------------------------------------

    if op in {
        "LOAD_CONST",
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "LOAD_FAST",
        "LOAD_DEREF",
        "PUSH_NULL",
        "LOAD_ATTR",
        "BINARY_OP",
        "UNARY_NOT",
        "UNARY_NEGATIVE",
        "UNARY_POSITIVE",
        "UNARY_INVERT",
        "COMPARE_OP",
        "IS_OP",
        "CONTAINS_OP",
        "CALL",
        "BUILD_LIST",
        "BUILD_TUPLE",
        "BUILD_SET",
        "BUILD_MAP",
        "BUILD_CONST_KEY_MAP",
        "BUILD_SLICE",
        "BINARY_SUBSCR",
        "COPY",
        "SWAP",
    }:
        from .stack import apply_instruction

        try:
            apply_instruction(
                stack,
                instruction,
            )
        except StackError as error:
            raise StatementError(
                f"Could not process {op}: {error}"
            ) from error

        return None

    # ------------------------------------------------------------------
    # Expression statement
    # ------------------------------------------------------------------

    if op == "POP_TOP":
        expression = _pop_expression(
            stack
        )

        return ExpressionStatement(
            expression=expression,
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Simple stores
    # ------------------------------------------------------------------

    if op in {
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_FAST",
        "STORE_DEREF",
    }:
        value = _pop_expression(
            stack
        )

        name = _require_name(
            instruction
        )

        return Assign(
            target=Name(
                name=name
            ),
            value=value,
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Delete names
    # ------------------------------------------------------------------

    if op in {
        "DELETE_NAME",
        "DELETE_GLOBAL",
        "DELETE_FAST",
        "DELETE_DEREF",
    }:
        name = _require_name(
            instruction
        )

        return Delete(
            target=Name(
                name=name
            ),
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------

    if op == "RETURN_VALUE":
        value = _pop_expression(
            stack
        )

        return Return(
            value=value,
            offset=instruction.offset,
        )

    if op == "RETURN_CONST":
        return Return(
            value=_consume_return_const(
                instruction
            ),
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Attribute stores
    # ------------------------------------------------------------------

    if op == "STORE_ATTR":
        from .ir import Attribute

        value = _pop_expression(
            stack
        )

        object_expression = _pop_expression(
            stack
        )

        name = _require_name(
            instruction
        )

        return Assign(
            target=Attribute(
                value=object_expression,
                name=name,
            ),
            value=value,
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Attribute delete
    # ------------------------------------------------------------------

    if op == "DELETE_ATTR":
        from .ir import Attribute

        object_expression = _pop_expression(
            stack
        )

        name = _require_name(
            instruction
        )

        return Delete(
            target=Attribute(
                value=object_expression,
                name=name,
            ),
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Subscript store
    # ------------------------------------------------------------------

    if op == "STORE_SUBSCR":
        from .ir import Subscript

        value = _pop_expression(
            stack
        )

        index = _pop_expression(
            stack
        )

        object_expression = _pop_expression(
            stack
        )

        return Assign(
            target=Subscript(
                value=object_expression,
                index=index,
            ),
            value=value,
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Subscript delete
    # ------------------------------------------------------------------

    if op == "DELETE_SUBSCR":
        from .ir import Subscript

        index = _pop_expression(
            stack
        )

        object_expression = _pop_expression(
            stack
        )

        return Delete(
            target=Subscript(
                value=object_expression,
                index=index,
            ),
            offset=instruction.offset,
        )

    # ------------------------------------------------------------------
    # Jumps and control-flow instructions
    # ------------------------------------------------------------------

    if op in {
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP_ABSOLUTE",
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
        "FOR_ITER",
        "GET_ITER",
        "END_FOR",
    }:
        # These instructions are intentionally left to the CFG/loop
        # reconstruction passes.
        return None

    # ------------------------------------------------------------------
    # Exception machinery
    # ------------------------------------------------------------------

    if op in {
        "RAISE_VARARGS",
        "RERAISE",
        "PUSH_EXC_INFO",
        "POP_EXCEPT",
        "CHECK_EXC_MATCH",
        "WITH_EXCEPT_START",
    }:
        return None

    # ------------------------------------------------------------------
    # Unsupported instruction
    # ------------------------------------------------------------------

    raise StatementError(
        f"Unsupported statement instruction: {op}"
    )


# ---------------------------------------------------------------------------
# Sequence processing
# ---------------------------------------------------------------------------


def reconstruct_statements(
    instructions: list[TInstruction],
) -> StatementResult:
    """
    Reconstruct statements from a linear instruction sequence.

    This function is deliberately conservative. Instructions that
    require control-flow analysis are left for later passes.
    """

    stack = VirtualStack()

    statements: list[IRStatement] = []

    for instruction in instructions:
        statement = process_instruction(
            stack,
            instruction,
        )

        if statement is not None:
            statements.append(
                statement
            )

    return StatementResult(
        statements=statements,
        stack=stack,
    )


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def render_statement(
    statement: IRStatement,
) -> str:
    """Render one IR statement as Python source."""

    if isinstance(
        statement,
        Assign,
    ):
        return (
            f"{render_expression(statement.target)}"
            f" = "
            f"{render_expression(statement.value)}"
        )

    if isinstance(
        statement,
        Delete,
    ):
        return (
            f"del "
            f"{render_expression(statement.target)}"
        )

    if isinstance(
        statement,
        ExpressionStatement,
    ):
        return render_expression(
            statement.expression
        )

    if isinstance(
        statement,
        Return,
    ):
        if statement.value is None:
            return "return"

        return (
            "return "
            + render_expression(
                statement.value
            )
        )

    if isinstance(
        statement,
        Pass,
    ):
        return "pass"

    raise StatementError(
        "Unsupported IR statement: "
        f"{type(statement).__name__}"
    )


def render_statements(
    statements: list[IRStatement],
    *,
    indent: int = 0,
    indent_text: str = "    ",
) -> str:
    """Render multiple statements."""

    prefix = indent_text * indent

    return "\n".join(
        prefix + render_statement(
            statement
        )
        for statement in statements
    )