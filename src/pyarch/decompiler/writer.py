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
    BoolOp,
    Break,
    Call,
    Class,
    Compare,
    Constant,
    Continue,
    Delete,
    DictComp,
    DictExpr,
    ExceptHandler,
    ExpressionStatement,
    For,
    FormattedValue,
    Function,
    GeneratorExp,
    If,
    Import,
    ImportFrom,
    IRExpression,
    IRStatement,
    JoinedStr,
    ListComp,
    ListExpr,
    Module,
    Name,
    Pass,
    Raise,
    Return,
    SetComp,
    SetExpr,
    SliceExpr,
    Starred,
    Subscript,
    Try,
    TupleExpr,
    UnaryOp,
    While,
    With,
    WithItem,
    Yield,
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


def _comprehension_clauses(
    generators: list,
) -> list[ast.comprehension]:
    return [
        ast.comprehension(
            target=target_to_ast(gen.target),
            iter=expression_to_ast(gen.iter),
            ifs=[expression_to_ast(cond) for cond in gen.ifs],
            is_async=0,
        )
        for gen in generators
    ]


def expression_to_ast(
    expression: IRExpression,
) -> ast.expr:
    """Convert an IR expression into Python AST."""

    if isinstance(expression, Yield):
        return ast.Yield(
            value=(
                expression_to_ast(expression.value)
                if expression.value is not None
                else None
            )
        )

    if isinstance(expression, BoolOp):
        return ast.BoolOp(
            op=ast.And() if expression.op == "and" else ast.Or(),
            values=[
                expression_to_ast(value)
                for value in expression.values
            ],
        )

    if isinstance(expression, JoinedStr):
        return ast.JoinedStr(
            values=[
                expression_to_ast(part)
                for part in expression.values
            ]
        )

    if isinstance(expression, FormattedValue):
        return ast.FormattedValue(
            value=expression_to_ast(expression.value),
            conversion=expression.conversion,
            format_spec=(
                ast.JoinedStr(
                    values=[
                        expression_to_ast(expression.format_spec)
                    ]
                )
                if expression.format_spec is not None
                else None
            ),
        )

    if isinstance(expression, ListComp):
        return ast.ListComp(
            elt=expression_to_ast(expression.element),
            generators=_comprehension_clauses(
                expression.generators
            ),
        )

    if isinstance(expression, SetComp):
        return ast.SetComp(
            elt=expression_to_ast(expression.element),
            generators=_comprehension_clauses(
                expression.generators
            ),
        )

    if isinstance(expression, DictComp):
        return ast.DictComp(
            key=expression_to_ast(expression.key),
            value=expression_to_ast(expression.value),
            generators=_comprehension_clauses(
                expression.generators
            ),
        )

    if isinstance(expression, GeneratorExp):
        return ast.GeneratorExp(
            elt=expression_to_ast(expression.element),
            generators=_comprehension_clauses(
                expression.generators
            ),
        )

    if isinstance(expression, Starred):
        return ast.Starred(
            value=expression_to_ast(expression.value),
            ctx=ast.Load(),
        )

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

    elif isinstance(node, ast.Starred):
        node.ctx = ast.Store()
        _set_store_context(node.value)

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

    if isinstance(
        statement,
        Break,
    ):
        return ast.Break()

    if isinstance(
        statement,
        If,
    ):
        return ast.If(
            test=expression_to_ast(statement.test),
            body=body_to_ast(statement.body),
            orelse=body_to_ast(
                statement.orelse, allow_empty=True
            ),
        )

    if isinstance(
        statement,
        While,
    ):
        return ast.While(
            test=expression_to_ast(statement.test),
            body=body_to_ast(statement.body),
            orelse=[],
        )

    if isinstance(
        statement,
        For,
    ):
        return ast.For(
            target=target_to_ast(statement.target),
            iter=expression_to_ast(statement.iter),
            body=body_to_ast(statement.body),
            orelse=[],
        )

    if isinstance(
        statement,
        Import,
    ):
        return ast.Import(
            names=[
                ast.alias(name=module, asname=asname)
                for module, asname in statement.names
            ]
        )

    if isinstance(
        statement,
        ImportFrom,
    ):
        return ast.ImportFrom(
            module=statement.module or None,
            names=[
                ast.alias(name=name, asname=asname)
                for name, asname in statement.names
            ],
            level=statement.level,
        )

    if isinstance(
        statement,
        Try,
    ):
        return ast.Try(
            body=body_to_ast(statement.body),
            handlers=[
                ast.ExceptHandler(
                    type=(
                        expression_to_ast(handler.type)
                        if handler.type is not None
                        else None
                    ),
                    name=handler.name,
                    body=body_to_ast(handler.body),
                )
                for handler in statement.handlers
            ],
            orelse=body_to_ast(
                statement.orelse, allow_empty=True
            ),
            finalbody=body_to_ast(
                statement.finalbody, allow_empty=True
            ),
        )

    if isinstance(statement, With):
        node_type = (
            ast.AsyncWith if statement.is_async else ast.With
        )

        return node_type(
            items=[
                ast.withitem(
                    context_expr=expression_to_ast(
                        item.context_expr
                    ),
                    optional_vars=(
                        target_to_ast(item.optional_vars)
                        if item.optional_vars is not None
                        else None
                    ),
                )
                for item in statement.items
            ],
            body=body_to_ast(statement.body),
        )

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


def body_to_ast(
    items: Iterable[object],
    *,
    allow_empty: bool = False,
) -> list[ast.stmt]:
    """
    Convert any body list (module/class/function/if/while/for) to
    AST statements, allowing nested Function/Class definitions to
    appear inline wherever ordinary statements can.

    `allow_empty` should be True only for an `if` statement's
    `orelse`, where an empty list is valid Python (no else clause) --
    every other body (function/class/while/for/if-body) must contain
    at least one statement.
    """

    result: list[ast.stmt] = []

    for item in items:
        if isinstance(item, Function):
            result.append(function_to_ast(item))
        elif isinstance(item, Class):
            result.append(class_to_ast(item))
        elif isinstance(item, IRStatement):
            result.append(statement_to_ast(item))
        else:
            raise WriterError(
                f"Unsupported body item: {type(item).__name__}"
            )

    if not result and not allow_empty:
        result = [ast.Pass()]

    return result


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

    defaults_by_name = getattr(
        function,
        "defaults",
        None,
    ) or {}

    annotations_by_name = getattr(
        function,
        "annotations",
        None,
    ) or {}

    def _make_arg(name: str) -> ast.arg:
        annotation = annotations_by_name.get(name)
        return ast.arg(
            arg=name,
            annotation=(
                expression_to_ast(annotation)
                if annotation is not None
                else None
            ),
        )

    args: list[ast.arg] = []
    kwonlyargs: list[ast.arg] = []
    vararg: ast.arg | None = None
    kwarg: ast.arg | None = None
    seen_star = False

    for name in parameters:
        if name == "*":
            seen_star = True
        elif name.startswith("**"):
            kwarg = _make_arg(name[2:])
        elif name.startswith("*"):
            vararg = _make_arg(name[1:])
            seen_star = True
        elif seen_star:
            kwonlyargs.append(_make_arg(name))
        else:
            args.append(_make_arg(name))

    # Positional defaults are right-aligned: a default for the Nth
    # positional arg from the end applies to `args[-N]`. Only a
    # trailing run (in original order) can have defaults in valid
    # Python; if defaults exist for some but not a trailing run,
    # that reflects a shape PyArch couldn't fully recover -- keep
    # only the trailing matched ones rather than emit invalid syntax.
    trailing_defaults: list[ast.expr] = []

    for arg in reversed(args):
        if arg.arg in defaults_by_name:
            trailing_defaults.insert(
                0, expression_to_ast(defaults_by_name[arg.arg])
            )
        else:
            break

    kw_defaults = [
        (
            expression_to_ast(defaults_by_name[arg.arg])
            if arg.arg in defaults_by_name
            else None
        )
        for arg in kwonlyargs
    ]

    return ast.arguments(
        posonlyargs=[],
        args=args,
        vararg=vararg,
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        kwarg=kwarg,
        defaults=trailing_defaults,
    )


def function_to_ast(
    function: Function,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    body = body_to_ast(
        function.body
    )

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
    body = body_to_ast(class_ir.body)

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
    body = body_to_ast(module.body)

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