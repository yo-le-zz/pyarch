from __future__ import annotations

"""
Python AST writer for PyArch.

The decompiler reconstructs an intermediate representation (IR).
This module converts that IR into Python's standard ``ast`` tree.

The final source code is generated with ``ast.unparse()``.

Architecture:

    PyArch IR
       ↓
    IR → Python AST
       ↓
    ast.fix_missing_locations()
       ↓
    ast.unparse()
       ↓
    .py
"""

import ast
from typing import Iterable

from .expressions import render_expression
from .ir import (
    Assign,
    Attribute,
    BinaryOp,
    Call,
    Class,
    Compare,
    Constant,
    Continue,
    Delete,
    DictExpr,
    ExpressionStatement,
    Function,
    IRExpression,
    IRNode,
    IRStatement,
    ListExpr,
    Module,
    Name,
    Pass,
    Raise,
    Return,
    SetExpr,
    SliceExpr,
    Subscript,
    TupleExpr,
    UnaryOp,
)


class WriterError(Exception):
    """Raised when IR cannot be converted to Python AST."""


# ---------------------------------------------------------------------------
# Expression writer
# ---------------------------------------------------------------------------


def _constant_to_ast(
    expression: Constant,
) -> ast.expr:
    """Convert an IR constant to an AST constant."""

    return ast.Constant(
        value=expression.value
    )


def _name_to_ast(
    expression: Name,
) -> ast.expr:
    """Convert an IR name to an AST name."""

    return ast.Name(
        id=expression.name,
        ctx=ast.Load(),
    )


def _attribute_to_ast(
    expression: Attribute,
) -> ast.expr:
    """Convert an IR attribute access."""

    return ast.Attribute(
        value=expression_to_ast(
            expression.value
        ),
        attr=expression.attribute,
        ctx=ast.Load(),
    )


def _subscript_to_ast(
    expression: Subscript,
) -> ast.expr:
    """Convert an IR subscript."""

    return ast.Subscript(
        value=expression_to_ast(
            expression.value
        ),
        slice=expression_to_ast(
            expression.slice
        ),
        ctx=ast.Load(),
    )


_BINARY_OPS: dict[str, type[ast.operator]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "@": ast.MatMult,
    "/": ast.Div,
    "//": ast.FloorDiv,
    "%": ast.Mod,
    "**": ast.Pow,
    "<<": ast.LShift,
    ">>": ast.RShift,
    "&": ast.BitAnd,
    "|": ast.BitOr,
    "^": ast.BitXor,
}


_UNARY_OPS: dict[str, type[ast.unaryop]] = {
    "+": ast.UAdd,
    "-": ast.USub,
    "~": ast.Invert,
    "not": ast.Not,
}


_COMPARE_OPS: dict[str, type[ast.cmpop]] = {
    "==": ast.Eq,
    "!=": ast.NotEq,
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
    "is": ast.Is,
    "is not": ast.IsNot,
    "in": ast.In,
    "not in": ast.NotIn,
}


def _binary_to_ast(
    expression: BinaryOp,
) -> ast.expr:
    """Convert a binary operation."""

    operator = _BINARY_OPS.get(
        expression.operator
    )

    if operator is None:
        raise WriterError(
            f"Unsupported binary operator: "
            f"{expression.operator!r}"
        )

    return ast.BinOp(
        left=expression_to_ast(
            expression.left
        ),
        op=operator(),
        right=expression_to_ast(
            expression.right
        ),
    )


def _unary_to_ast(
    expression: UnaryOp,
) -> ast.expr:
    """Convert a unary operation."""

    operator = _UNARY_OPS.get(
        expression.operator
    )

    if operator is None:
        raise WriterError(
            f"Unsupported unary operator: "
            f"{expression.operator!r}"
        )

    return ast.UnaryOp(
        op=operator(),
        operand=expression_to_ast(
            expression.operand
        ),
    )


def _compare_to_ast(
    expression: Compare,
) -> ast.expr:
    """Convert an IR comparison."""

    operators = [
        _COMPARE_OPS.get(
            operator
        )
        for operator in expression.operators
    ]

    if any(
        operator is None
        for operator in operators
    ):
        raise WriterError(
            "Unsupported comparison operator."
        )

    return ast.Compare(
        left=expression_to_ast(
            expression.left
        ),
        ops=[
            operator()
            for operator in operators
            if operator is not None
        ],
        comparators=[
            expression_to_ast(
                value
            )
            for value in expression.comparators
        ],
    )


def _call_to_ast(
    expression: Call,
) -> ast.expr:
    """Convert an IR function call."""

    return ast.Call(
        func=expression_to_ast(
            expression.function
        ),
        args=[
            expression_to_ast(
                argument
            )
            for argument in expression.arguments
        ],
        keywords=[],
    )


def _list_to_ast(
    expression: ListExpr,
) -> ast.expr:
    return ast.List(
        elts=[
            expression_to_ast(
                value
            )
            for value in expression.elements
        ],
        ctx=ast.Load(),
    )


def _tuple_to_ast(
    expression: TupleExpr,
) -> ast.expr:
    return ast.Tuple(
        elts=[
            expression_to_ast(
                value
            )
            for value in expression.elements
        ],
        ctx=ast.Load(),
    )


def _set_to_ast(
    expression: SetExpr,
) -> ast.expr:
    return ast.Set(
        elts=[
            expression_to_ast(
                value
            )
            for value in expression.elements
        ]
    )


def _dict_to_ast(
    expression: DictExpr,
) -> ast.expr:
    keys = [
        expression_to_ast(
            key
        )
        for key in expression.keys
    ]

    values = [
        expression_to_ast(
            value
        )
        for value in expression.values
    ]

    return ast.Dict(
        keys=keys,
        values=values,
    )


def _slice_to_ast(
    expression: SliceExpr,
) -> ast.expr:
    return ast.Slice(
        lower=(
            expression_to_ast(
                expression.lower
            )
            if expression.lower is not None
            else None
        ),
        upper=(
            expression_to_ast(
                expression.upper
            )
            if expression.upper is not None
            else None
        ),
        step=(
            expression_to_ast(
                expression.step
            )
            if expression.step is not None
            else None
        ),
    )


def expression_to_ast(
    expression: IRExpression,
) -> ast.expr:
    """
    Convert any supported IR expression to Python AST.

    This is the central expression dispatcher.
    """

    if isinstance(
        expression,
        Constant,
    ):
        return _constant_to_ast(
            expression
        )

    if isinstance(
        expression,
        Name,
    ):
        return _name_to_ast(
            expression
        )

    if isinstance(
        expression,
        Attribute,
    ):
        return _attribute_to_ast(
            expression
        )

    if isinstance(
        expression,
        Subscript,
    ):
        return _subscript_to_ast(
            expression
        )

    if isinstance(
        expression,
        BinaryOp,
    ):
        return _binary_to_ast(
            expression
        )

    if isinstance(
        expression,
        UnaryOp,
    ):
        return _unary_to_ast(
            expression
        )

    if isinstance(
        expression,
        Compare,
    ):
        return _compare_to_ast(
            expression
        )

    if isinstance(
        expression,
        Call,
    ):
        return _call_to_ast(
            expression
        )

    if isinstance(
        expression,
        ListExpr,
    ):
        return _list_to_ast(
            expression
        )

    if isinstance(
        expression,
        TupleExpr,
    ):
        return _tuple_to_ast(
            expression
        )

    if isinstance(
        expression,
        SetExpr,
    ):
        return _set_to_ast(
            expression
        )

    if isinstance(
        expression,
        DictExpr,
    ):
        return _dict_to_ast(
            expression
        )

    if isinstance(
        expression,
        SliceExpr,
    ):
        return _slice_to_ast(
            expression
        )

    raise WriterError(
        f"Unsupported IR expression: "
        f"{type(expression).__name__}"
    )


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


def _target_to_ast(
    expression: IRExpression,
) -> ast.expr:
    """
    Convert an assignment target.

    ``Name`` and ``Attribute``/``Subscript`` need Store context.
    """

    result = expression_to_ast(
        expression
    )

    if isinstance(
        result,
        (
            ast.Name,
            ast.Attribute,
            ast.Subscript,
        ),
    ):
        result.ctx = ast.Store()

    elif isinstance(
        result,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        result.ctx = ast.Store()

        for element in result.elts:
            if isinstance(
                element,
                (
                    ast.Name,
                    ast.Attribute,
                    ast.Subscript,
                    ast.Tuple,
                    ast.List,
                ),
            ):
                element.ctx = ast.Store()

    return result


# ---------------------------------------------------------------------------
# Statement writer
# ---------------------------------------------------------------------------


def _expression_statement_to_ast(
    statement: ExpressionStatement,
) -> ast.stmt:
    return ast.Expr(
        value=expression_to_ast(
            statement.expression
        )
    )


def _assign_to_ast(
    statement: Assign,
) -> ast.stmt:
    return ast.Assign(
        targets=[
            _target_to_ast(
                target
            )
            for target in statement.targets
        ],
        value=expression_to_ast(
            statement.value
        ),
    )


def _delete_to_ast(
    statement: Delete,
) -> ast.stmt:
    return ast.Delete(
        targets=[
            _target_to_ast(
                target
            )
            for target in statement.targets
        ]
    )


def _return_to_ast(
    statement: Return,
) -> ast.stmt:
    value = getattr(
        statement,
        "value",
        None,
    )

    return ast.Return(
        value=(
            expression_to_ast(value)
            if value is not None
            else None
        )
    )


def _raise_to_ast(
    statement: Raise,
) -> ast.stmt:
    value = getattr(
        statement,
        "exception",
        None,
    )

    cause = getattr(
        statement,
        "cause",
        None,
    )

    return ast.Raise(
        exc=(
            expression_to_ast(value)
            if value is not None
            else None
        ),
        cause=(
            expression_to_ast(cause)
            if cause is not None
            else None
        ),
    )


def statement_to_ast(
    statement: IRStatement,
) -> ast.stmt:
    """Convert one IR statement to Python AST."""

    if isinstance(
        statement,
        ExpressionStatement,
    ):
        return _expression_statement_to_ast(
            statement
        )

    if isinstance(
        statement,
        Assign,
    ):
        return _assign_to_ast(
            statement
        )

    if isinstance(
        statement,
        Delete,
    ):
        return _delete_to_ast(
            statement
        )

    if isinstance(
        statement,
        Return,
    ):
        return _return_to_ast(
            statement
        )

    if isinstance(
        statement,
        Raise,
    ):
        return _raise_to_ast(
            statement
        )

    if isinstance(
        statement,
        Pass,
    ):
        return ast.Pass()

    if isinstance(
        statement,
        Continue,
    ):
        return ast.Continue()

    raise WriterError(
        f"Unsupported IR statement: "
        f"{type(statement).__name__}"
    )


def statements_to_ast(
    statements: Iterable[IRStatement],
) -> list[ast.stmt]:
    """Convert multiple IR statements."""

    result: list[ast.stmt] = []

    for statement in statements:
        result.append(
            statement_to_ast(
                statement
            )
        )

    return result


# ---------------------------------------------------------------------------
# Function writer
# ---------------------------------------------------------------------------


def _function_arguments(
    function: Function,
) -> ast.arguments:
    """
    Build a basic ``ast.arguments`` object.

    Defaults and annotations will be recovered later from MAKE_FUNCTION
    metadata.
    """

    parameters = getattr(
        function,
        "parameters",
        [],
    )

    args = [
        ast.arg(
            arg=name
        )
        for name in parameters
    ]

    return ast.arguments(
        posonlyargs=[],
        args=args,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def function_to_ast(
    function: Function,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Convert a PyArch Function to Python AST."""

    body = statements_to_ast(
        getattr(
            function,
            "body",
            [],
        )
    )

    if not body:
        body = [
            ast.Pass()
        ]

    node_type = (
        ast.AsyncFunctionDef
        if getattr(
            function,
            "is_async",
            False,
        )
        else ast.FunctionDef
    )

    return node_type(
        name=function.name,
        args=_function_arguments(
            function
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )


# ---------------------------------------------------------------------------
# Class writer
# ---------------------------------------------------------------------------


def class_to_ast(
    class_ir: Class,
) -> ast.ClassDef:
    """Convert a PyArch Class to Python AST."""

    body = statements_to_ast(
        getattr(
            class_ir,
            "body",
            [],
        )
    )

    if not body:
        body = [
            ast.Pass()
        ]

    bases = [
        expression_to_ast(
            base
        )
        for base in getattr(
            class_ir,
            "bases",
            [],
        )
    ]

    decorators = [
        expression_to_ast(
            decorator
        )
        for decorator in getattr(
            class_ir,
            "decorators",
            [],
        )
    ]

    return ast.ClassDef(
        name=class_ir.name,
        bases=bases,
        keywords=[],
        body=body,
        decorator_list=decorators,
    )


# ---------------------------------------------------------------------------
# Module writer
# ---------------------------------------------------------------------------


def module_to_ast(
    module: Module,
) -> ast.Module:
    """Convert a PyArch Module to Python AST."""

    body = statements_to_ast(
        getattr(
            module,
            "body",
            [],
        )
    )

    return ast.Module(
        body=body,
        type_ignores=[],
    )


def fix_locations(
    tree: ast.AST,
) -> ast.AST:
    """Populate missing AST source locations."""

    return ast.fix_missing_locations(
        tree
    )


def unparse(
    tree: ast.AST,
) -> str:
    """Convert Python AST to source code."""

    return ast.unparse(
        tree
    )


def write_module(
    module: Module,
) -> str:
    """
    Convert a PyArch module directly to Python source.
    """

    tree = module_to_ast(
        module
    )

    fix_locations(
        tree
    )

    return unparse(
        tree
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_source(
    source: str,
) -> ast.Module:
    """
    Parse generated Python source to make sure it is syntactically
    valid.
    """

    try:
        tree = ast.parse(
            source
        )
    except SyntaxError as error:
        raise WriterError(
            f"Generated invalid Python: {error}"
        ) from error

    return tree


def write_and_validate(
    module: Module,
) -> str:
    """Generate Python source and validate it."""

    source = write_module(
        module
    )

    validate_source(
        source
    )

    return source


__all__ = [
    "WriterError",
    "expression_to_ast",
    "statement_to_ast",
    "statements_to_ast",
    "function_to_ast",
    "class_to_ast",
    "module_to_ast",
    "fix_locations",
    "unparse",
    "write_module",
    "validate_source",
    "write_and_validate",
]