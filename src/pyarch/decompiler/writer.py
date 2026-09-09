from __future__ import annotations

"""
Python AST writer for PyArch.

Converts PyArch's intermediate representation into Python's standard
AST and finally into valid Python source using ast.unparse().
"""

import ast
from typing import Iterable

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
    """Raised when the IR cannot be converted to Python AST."""


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _constant_to_ast(
    expression: Constant,
) -> ast.expr:
    return ast.Constant(
        value=expression.value
    )


def _name_to_ast(
    expression: Name,
) -> ast.expr:
    return ast.Name(
        id=expression.name,
        ctx=ast.Load(),
    )


def _attribute_to_ast(
    expression: Attribute,
) -> ast.expr:
    return ast.Attribute(
        value=expression_to_ast(
            expression.value
        ),
        attr=expression.name,
        ctx=ast.Load(),
    )


def _subscript_to_ast(
    expression: Subscript,
) -> ast.expr:
    return ast.Subscript(
        value=expression_to_ast(
            expression.value
        ),
        slice=expression_to_ast(
            expression.index
        ),
        ctx=ast.Load(),
    )


def _binary_to_ast(
    expression: BinaryOp,
) -> ast.expr:
    operator = _BINARY_OPS.get(
        expression.op
    )

    if operator is None:
        raise WriterError(
            f"Unsupported binary operator: "
            f"{expression.op!r}"
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
    operator = _UNARY_OPS.get(
        expression.op
    )

    if operator is None:
        raise WriterError(
            f"Unsupported unary operator: "
            f"{expression.op!r}"
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
    operator = _COMPARE_OPS.get(
        expression.op
    )

    if operator is None:
        raise WriterError(
            f"Unsupported comparison operator: "
            f"{expression.op!r}"
        )

    return ast.Compare(
        left=expression_to_ast(
            expression.left
        ),
        ops=[
            operator()
        ],
        comparators=[
            expression_to_ast(
                expression.right
            )
        ],
    )


def _call_to_ast(
    expression: Call,
) -> ast.expr:
    keywords: list[ast.keyword] = []

    for name, value in expression.keywords:
        keywords.append(
            ast.keyword(
                arg=name,
                value=expression_to_ast(
                    value
                ),
            )
        )

    return ast.Call(
        func=expression_to_ast(
            expression.function
        ),
        args=[
            expression_to_ast(
                argument
            )
            for argument in expression.args
        ],
        keywords=keywords,
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
    keys: list[ast.expr | None] = []
    values: list[ast.expr] = []

    for key, value in expression.entries:
        keys.append(
            (
                expression_to_ast(key)
                if key is not None
                else None
            )
        )

        values.append(
            expression_to_ast(
                value
            )
        )

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
    """Convert an IR expression into Python AST."""

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
# Assignment targets
# ---------------------------------------------------------------------------


def _set_store_context(
    node: ast.expr,
) -> ast.expr:
    """Convert an expression node into an assignment target."""

    if isinstance(
        node,
        (
            ast.Name,
            ast.Attribute,
            ast.Subscript,
        ),
    ):
        node.ctx = ast.Store()

    elif isinstance(
        node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        node.ctx = ast.Store()

        for element in node.elts:
            _set_store_context(
                element
            )

    return node


def target_to_ast(
    expression: IRExpression,
) -> ast.expr:
    return _set_store_context(
        expression_to_ast(
            expression
        )
    )


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


def _expression_statement_to_ast(
    statement: ExpressionStatement,
) -> ast.Expr:
    return ast.Expr(
        value=expression_to_ast(
            statement.expression
        )
    )


def _assign_to_ast(
    statement: Assign,
) -> ast.Assign:
    return ast.Assign(
        targets=[
            target_to_ast(
                statement.target
            )
        ],
        value=expression_to_ast(
            statement.value
        ),
    )


def _delete_to_ast(
    statement: Delete,
) -> ast.Delete:
    return ast.Delete(
        targets=[
            target_to_ast(
                statement.target
            )
        ]
    )


def _return_to_ast(
    statement: Return,
) -> ast.Return:
    return ast.Return(
        value=(
            expression_to_ast(
                statement.value
            )
            if statement.value is not None
            else None
        )
    )


def _raise_to_ast(
    statement: Raise,
) -> ast.Raise:
    return ast.Raise(
        exc=(
            expression_to_ast(
                statement.exception
            )
            if statement.exception is not None
            else None
        ),
        cause=(
            expression_to_ast(
                statement.cause
            )
            if statement.cause is not None
            else None
        ),
    )


def statement_to_ast(
    statement: IRStatement,
) -> ast.stmt:
    """Convert one IR statement into Python AST."""

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
    return [
        statement_to_ast(
            statement
        )
        for statement in statements
    ]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _function_arguments(
    function: Function,
) -> ast.arguments:
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
    body = statements_to_ast(
        function.body
    )

    if not body:
        body = [
            ast.Pass()
        ]

    node_type = (
        ast.AsyncFunctionDef
        if function.is_async
        else ast.FunctionDef
    )

    return node_type(
        name=function.name,
        args=_function_arguments(
            function
        ),
        body=body,
        decorator_list=[
            expression_to_ast(
                decorator
            )
            for decorator in function.decorators
        ],
        returns=(
            expression_to_ast(
                function.returns
            )
            if function.returns is not None
            else None
        ),
        type_comment=None,
    )


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


def class_to_ast(
    class_ir: Class,
) -> ast.ClassDef:
    body: list[ast.stmt] = []

    for item in class_ir.body:
        if isinstance(
            item,
            Function,
        ):
            body.append(
                function_to_ast(
                    item
                )
            )
        elif isinstance(
            item,
            IRStatement,
        ):
            body.append(
                statement_to_ast(
                    item
                )
            )
        else:
            raise WriterError(
                f"Unsupported class body item: "
                f"{type(item).__name__}"
            )

    if not body:
        body = [
            ast.Pass()
        ]

    return ast.ClassDef(
        name=class_ir.name,
        bases=[
            expression_to_ast(
                base
            )
            for base in class_ir.bases
        ],
        keywords=[],
        body=body,
        decorator_list=[
            expression_to_ast(
                decorator
            )
            for decorator in class_ir.decorators
        ],
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def module_to_ast(
    module: Module,
) -> ast.Module:
    body: list[ast.stmt] = []

    for item in module.body:
        if isinstance(
            item,
            Function,
        ):
            body.append(
                function_to_ast(
                    item
                )
            )

        elif isinstance(
            item,
            Class,
        ):
            body.append(
                class_to_ast(
                    item
                )
            )

        elif isinstance(
            item,
            IRStatement,
        ):
            body.append(
                statement_to_ast(
                    item
                )
            )

        else:
            raise WriterError(
                f"Unsupported module body item: "
                f"{type(item).__name__}"
            )

    return ast.Module(
        body=body,
        type_ignores=[],
    )


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def fix_locations(
    tree: ast.AST,
) -> ast.AST:
    return ast.fix_missing_locations(
        tree
    )


def unparse(
    tree: ast.AST,
) -> str:
    return ast.unparse(
        tree
    )


def write_module(
    module: Module,
) -> str:
    """Convert a PyArch Module into Python source."""

    tree = module_to_ast(
        module
    )

    fix_locations(
        tree
    )

    return unparse(
        tree
    )


def validate_source(
    source: str,
) -> ast.Module:
    """Verify that generated source is valid Python."""

    try:
        return ast.parse(
            source
        )
    except SyntaxError as error:
        raise WriterError(
            f"Generated invalid Python: {error}"
        ) from error


def write_and_validate(
    module: Module,
) -> str:
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
    "target_to_ast",
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