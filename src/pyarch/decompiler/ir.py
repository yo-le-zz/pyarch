from __future__ import annotations

"""
PyArch Intermediate Representation.

This module defines the version-independent intermediate representation
used by the real Python decompiler.

Pipeline:

    CPython bytecode
          ↓
    translate.py
          ↓
    TInstruction
          ↓
    IR
          ↓
    CFG / stack / expressions / statements
          ↓
    Python source
"""


from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base nodes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IRNode:
    """Base class for every PyArch IR node."""

    offset: int | None = None


@dataclass(slots=True)
class IRInstruction(IRNode):
    """
    Generic IR instruction.

    This is useful for instructions that have already been normalized
    by translate.py but do not yet have a specialized IR representation.
    """

    op: str = ""
    arg: int | None = None
    value: Any = None
    target: int | None = None


# ---------------------------------------------------------------------------
# Values / expressions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IRExpression(IRNode):
    """Base class for expressions."""

    pass


@dataclass(slots=True)
class Constant(IRExpression):
    """A Python constant."""

    value: Any = None


@dataclass(slots=True)
class Name(IRExpression):
    """A Python variable/name reference."""

    name: str = ""


@dataclass(slots=True)
class Attribute(IRExpression):
    """An attribute access: obj.attribute."""

    value: IRExpression = field(
        default_factory=IRExpression
    )
    name: str = ""


@dataclass(slots=True)
class Subscript(IRExpression):
    """A subscription: value[index]."""

    value: IRExpression = field(
        default_factory=IRExpression
    )
    index: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class UnaryOp(IRExpression):
    """A unary operation such as -x, +x, ~x or not x."""

    op: str = ""

    operand: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class BinaryOp(IRExpression):
    """A binary operation such as x + y or x * y."""

    op: str = ""

    left: IRExpression = field(
        default_factory=IRExpression
    )

    right: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class Compare(IRExpression):
    """A comparison such as x == y or x < y."""

    op: str = ""

    left: IRExpression = field(
        default_factory=IRExpression
    )

    right: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class Call(IRExpression):
    """A Python function/method call."""

    function: IRExpression = field(
        default_factory=IRExpression
    )

    args: list[IRExpression] = field(
        default_factory=list
    )

    keywords: list[tuple[str | None, IRExpression]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ListExpr(IRExpression):
    """A list literal."""

    elements: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class TupleExpr(IRExpression):
    """A tuple literal."""

    elements: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class SetExpr(IRExpression):
    """A set literal."""

    elements: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class DictExpr(IRExpression):
    """A dictionary literal."""

    entries: list[
        tuple[IRExpression | None, IRExpression]
    ] = field(
        default_factory=list
    )


@dataclass(slots=True)
class SliceExpr(IRExpression):
    """A slice expression."""

    lower: IRExpression | None = None
    upper: IRExpression | None = None
    step: IRExpression | None = None


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IRStatement(IRNode):
    """Base class for statements."""

    pass


@dataclass(slots=True)
class ExpressionStatement(IRStatement):
    """An expression used as a statement."""

    expression: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class Assign(IRStatement):
    """A simple assignment."""

    target: IRExpression = field(
        default_factory=IRExpression
    )

    value: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class Delete(IRStatement):
    """A deletion such as del x or del obj.attr."""

    target: IRExpression = field(
        default_factory=IRExpression
    )


@dataclass(slots=True)
class Return(IRStatement):
    """A return statement."""

    value: IRExpression | None = None


@dataclass(slots=True)
class Raise(IRStatement):
    """A raise statement."""

    exception: IRExpression | None = None
    cause: IRExpression | None = None


@dataclass(slots=True)
class Import(IRStatement):
    """An `import a, b.c, d as e` statement."""

    names: list = field(
        default_factory=list
    )


@dataclass(slots=True)
class ImportFrom(IRStatement):
    """A `from module import a, b as c` statement."""

    module: str = ""

    names: list = field(
        default_factory=list
    )

    level: int = 0


@dataclass(slots=True)
class Pass(IRStatement):
    """A Python pass statement."""

    pass


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Jump(IRStatement):
    """An unconditional jump."""

    target: int = 0


@dataclass(slots=True)
class ConditionalJump(IRStatement):
    """
    A conditional branch.

    The condition is evaluated and control flow goes to either
    true_target or false_target.
    """

    condition: IRExpression = field(
        default_factory=IRExpression
    )

    true_target: int = 0
    false_target: int = 0


@dataclass(slots=True)
class Break(IRStatement):
    """A reconstructed break statement."""

    pass


@dataclass(slots=True)
class Continue(IRStatement):
    """A reconstructed continue statement."""

    pass


@dataclass(slots=True)
class If(IRStatement):
    """A structured if/elif/else statement."""

    test: IRExpression = field(
        default_factory=IRExpression
    )

    body: list = field(
        default_factory=list
    )

    orelse: list = field(
        default_factory=list
    )


@dataclass(slots=True)
class While(IRStatement):
    """A structured while loop."""

    test: IRExpression = field(
        default_factory=IRExpression
    )

    body: list = field(
        default_factory=list
    )


@dataclass(slots=True)
class For(IRStatement):
    """A structured for loop: for target in iter: body."""

    target: IRExpression = field(
        default_factory=IRExpression
    )

    iter: IRExpression = field(
        default_factory=IRExpression
    )

    body: list = field(
        default_factory=list
    )


@dataclass(slots=True)
class ExceptHandler(IRNode):
    """One `except [Type [as name]]:` clause."""

    type: IRExpression | None = None
    name: str | None = None
    body: list = field(default_factory=list)


@dataclass(slots=True)
class Try(IRStatement):
    """A `try`/`except`/`finally` statement."""

    body: list = field(default_factory=list)
    handlers: list = field(default_factory=list)
    orelse: list = field(default_factory=list)
    finalbody: list = field(default_factory=list)


@dataclass(slots=True)
class Starred(IRExpression):
    """A `*name` target in tuple unpacking, e.g. `a, *rest = seq`."""

    value: IRExpression = field(default_factory=IRExpression)


@dataclass(slots=True)
class WithItem(IRNode):
    context_expr: IRExpression = field(default_factory=IRExpression)
    optional_vars: IRExpression | None = None


@dataclass(slots=True)
class With(IRStatement):
    items: list = field(default_factory=list)
    body: list = field(default_factory=list)
    is_async: bool = False


@dataclass(slots=True)
class FormattedValue(IRExpression):
    """One `{expr!conv:spec}` component of an f-string."""

    value: IRExpression = field(default_factory=IRExpression)
    conversion: int = -1
    format_spec: IRExpression | None = None


@dataclass(slots=True)
class JoinedStr(IRExpression):
    """An f-string: a sequence of literal and formatted parts."""

    values: list = field(default_factory=list)


@dataclass(slots=True)
class BoolOp(IRExpression):
    """A short-circuit `and`/`or` expression (not statement-level if)."""

    op: str = "and"
    values: list = field(default_factory=list)


@dataclass(slots=True)
class Yield(IRExpression):
    value: IRExpression | None = None


@dataclass(slots=True)
class YieldFrom(IRExpression):
    value: IRExpression = field(default_factory=IRExpression)


@dataclass(slots=True)
class Comprehension(IRNode):
    """One `for target in iter [if cond]...` clause of a comprehension."""

    target: IRExpression = field(default_factory=IRExpression)
    iter: IRExpression = field(default_factory=IRExpression)
    ifs: list = field(default_factory=list)


@dataclass(slots=True)
class ListComp(IRExpression):
    element: IRExpression = field(default_factory=IRExpression)
    generators: list = field(default_factory=list)


@dataclass(slots=True)
class SetComp(IRExpression):
    element: IRExpression = field(default_factory=IRExpression)
    generators: list = field(default_factory=list)


@dataclass(slots=True)
class DictComp(IRExpression):
    key: IRExpression = field(default_factory=IRExpression)
    value: IRExpression = field(default_factory=IRExpression)
    generators: list = field(default_factory=list)


@dataclass(slots=True)
class GeneratorExp(IRExpression):
    element: IRExpression = field(default_factory=IRExpression)
    generators: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BasicBlock(IRNode):
    """
    A basic block in the control-flow graph.

    Instructions are kept here before they are transformed into
    higher-level statements.
    """

    block_id: int = 0

    instructions: list[IRInstruction] = field(
        default_factory=list
    )

    successors: list[int] = field(
        default_factory=list
    )

    predecessors: list[int] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ControlFlowGraph(IRNode):
    """Control-flow graph for a code object."""

    blocks: list[BasicBlock] = field(
        default_factory=list
    )

    entry: int | None = None


# ---------------------------------------------------------------------------
# Functions / classes / modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Function(IRNode):
    """A reconstructed Python function."""

    name: str = ""

    parameters: list[str] = field(
        default_factory=list
    )

    defaults: dict = field(
        default_factory=dict
    )

    annotations: dict = field(
        default_factory=dict
    )

    body: list[IRStatement] = field(
        default_factory=list
    )

    decorators: list[IRExpression] = field(
        default_factory=list
    )

    returns: IRExpression | None = None

    is_async: bool = False

    is_generator: bool = False


@dataclass(slots=True)
class Class(IRNode):
    """A reconstructed Python class."""

    name: str = ""

    bases: list[IRExpression] = field(
        default_factory=list
    )

    body: list[IRStatement | Function] = field(
        default_factory=list
    )

    decorators: list[IRExpression] = field(
        default_factory=list
    )


@dataclass(slots=True)
class Module(IRNode):
    """A complete reconstructed Python module."""

    name: str = ""

    filename: str = ""

    body: list[
        IRStatement
        | Function
        | Class
    ] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_expression(
    node: IRNode,
) -> bool:
    """Return True if an IR node represents an expression."""

    return isinstance(
        node,
        IRExpression,
    )


def is_statement(
    node: IRNode,
) -> bool:
    """Return True if an IR node represents a statement."""

    return isinstance(
        node,
        IRStatement,
    )


def walk_ir(
    node: Any,
):
    """
    Recursively walk an IR tree.

    This is intentionally generic so later analysis passes can inspect
    every reconstructed node without knowing its exact type.
    """

    if isinstance(node, IRNode):
        yield node

    if isinstance(node, list):
        for item in node:
            yield from walk_ir(item)

        return

    if isinstance(node, tuple):
        for item in node:
            yield from walk_ir(item)

        return

    if not isinstance(node, IRNode):
        return

    for value in vars(node).values():
        yield from walk_ir(value)