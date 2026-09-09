from __future__ import annotations

"""
Class reconstruction for PyArch.

This module extracts class-related information from Python code
objects and prepares it for the common PyArch IR.

The bytecode -> AST conversion is intentionally left to the later
decompiler engine.
"""

from dataclasses import dataclass
from types import CodeType

from .ir import (
    Class,
    Function,
    IRExpression,
    IRStatement,
    Name,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClassError(Exception):
    """Raised when class reconstruction fails."""


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassMetadata:
    """Metadata recovered for a Python class."""

    name: str

    bases: list[IRExpression]

    keywords: list[tuple[str, IRExpression]]

    decorators: list[IRExpression]

    methods: list[Function]

    body: list[IRStatement]


# ---------------------------------------------------------------------------
# Code object classification
# ---------------------------------------------------------------------------


def is_class_body(
    code: CodeType,
) -> bool:
    """
    Return whether a code object looks like a class body.

    Python class bodies are compiled as separate code objects whose
    name normally corresponds to the class name.

    This function is intentionally conservative because bytecode alone
    does not always preserve the complete source-level context.
    """

    if code.co_name == "<module>":
        return False

    return True


def nested_code_objects(
    code: CodeType,
) -> list[CodeType]:
    """Return code objects directly contained in a code object."""

    result: list[CodeType] = []

    for constant in code.co_consts:
        if isinstance(
            constant,
            CodeType,
        ):
            result.append(constant)

    return result


# ---------------------------------------------------------------------------
# Function classification
# ---------------------------------------------------------------------------


def _looks_like_function(
    code: CodeType,
) -> bool:
    """
    Determine whether a nested code object likely represents a method
    or normal function.

    Compiler-generated comprehensions and lambdas are excluded here;
    they are handled by other reconstruction passes.
    """

    if code.co_name in {
        "<listcomp>",
        "<setcomp>",
        "<dictcomp>",
        "<genexpr>",
        "<lambda>",
    }:
        return False

    return code.co_name != "<module>"


def discover_methods(
    code: CodeType,
) -> list[Function]:
    """
    Discover method-like nested code objects.

    Function bodies are populated later by the main decompiler engine.
    """

    from .functions import build_function

    methods: list[Function] = []

    for nested in nested_code_objects(
        code
    ):
        if not _looks_like_function(
            nested
        ):
            continue

        methods.append(
            build_function(
                nested
            )
        )

    return methods


# ---------------------------------------------------------------------------
# Special methods
# ---------------------------------------------------------------------------


_SPECIAL_METHODS = {
    "__init__",
    "__new__",
    "__del__",
    "__repr__",
    "__str__",
    "__bytes__",
    "__format__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__bool__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__call__",
    "__len__",
    "__iter__",
    "__next__",
    "__contains__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
}


def is_special_method(
    function: Function,
) -> bool:
    """Return whether a function is a Python special method."""

    return function.name in _SPECIAL_METHODS


# ---------------------------------------------------------------------------
# Class construction
# ---------------------------------------------------------------------------


def build_class(
    code: CodeType,
    *,
    bases: list[IRExpression] | None = None,
    keywords: list[tuple[str, IRExpression]] | None = None,
    decorators: list[IRExpression] | None = None,
    body: list[IRStatement] | None = None,
) -> Class:
    """
    Build a Class IR node from a class-body code object.

    Base classes, metaclass keywords and decorators are supplied by
    the surrounding MAKE_FUNCTION / LOAD_BUILD_CLASS analysis.
    """

    methods = discover_methods(
        code
    )

    return Class(
        offset=None,
        name=code.co_name,
        bases=[] if bases is None else bases,
        keywords=(
            []
            if keywords is None
            else keywords
        ),
        body=[] if body is None else body,
        methods=methods,
        decorators=(
            []
            if decorators is None
            else decorators
        ),
    )


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------


def inspect_class(
    code: CodeType,
) -> ClassMetadata:
    """
    Inspect a class-body code object.

    This provides a convenient metadata representation for the
    decompiler engine.
    """

    class_ir = build_class(
        code
    )

    return ClassMetadata(
        name=class_ir.name,
        bases=list(
            class_ir.bases
        ),
        keywords=list(
            class_ir.keywords
        ),
        decorators=list(
            class_ir.decorators
        ),
        methods=list(
            class_ir.methods
        ),
        body=list(
            class_ir.body
        ),
    )


# ---------------------------------------------------------------------------
# Method lookup
# ---------------------------------------------------------------------------


def find_method(
    class_ir: Class,
    name: str,
) -> Function | None:
    """Find a method by name."""

    for method in class_ir.methods:
        if method.name == name:
            return method

    return None


def find_special_methods(
    class_ir: Class,
) -> list[Function]:
    """Return all special methods defined by the class."""

    return [
        method
        for method in class_ir.methods
        if is_special_method(
            method
        )
    ]


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


def class_attribute_names(
    code: CodeType,
) -> list[str]:
    """
    Recover names referenced by a class body.

    This is only a preliminary view. The engine will later distinguish
    actual assignments from names merely loaded by the class body.
    """

    return sorted(
        {
            name
            for name in code.co_names
            if isinstance(name, str)
            and name.isidentifier()
        }
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_class_signature(
    class_ir: Class,
) -> str:
    """Render a class declaration."""

    bases = ", ".join(
        _render_expression(base)
        for base in class_ir.bases
    )

    keywords = ", ".join(
        f"{name}={_render_expression(value)}"
        for name, value in class_ir.keywords
    )

    arguments = ", ".join(
        part
        for part in (
            bases,
            keywords,
        )
        if part
    )

    if arguments:
        return (
            f"class {class_ir.name}"
            f"({arguments})"
        )

    return f"class {class_ir.name}"


def _render_expression(
    expression: IRExpression,
) -> str:
    """Render an IR expression for debugging."""

    from .expressions import render_expression

    return render_expression(
        expression
    ).text


def render_class(
    class_ir: Class,
    *,
    indent: int = 0,
) -> str:
    """
    Render a class using the current IR.

    This is a temporary/debug renderer. The final source writer will
    use Python's `ast` module.
    """

    prefix = "    " * indent

    signature = (
        prefix
        + render_class_signature(
            class_ir
        )
        + ":"
    )

    lines = [signature]

    if class_ir.body:
        from .statements import render_statements

        body = render_statements(
            class_ir.body,
            indent=indent + 1,
        )

        lines.append(body)

    for method in class_ir.methods:
        from .functions import render_function

        lines.append(
            render_function(
                method,
                indent=indent + 1,
            )
        )

    if len(lines) == 1:
        lines.append(
            prefix
            + "    pass"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Class discovery
# ---------------------------------------------------------------------------


def discover_classes(
    code: CodeType,
) -> list[Class]:
    """
    Discover class-like code objects directly nested inside `code`.

    Full class construction is completed later when the surrounding
    bytecode reveals BUILD_CLASS arguments, bases and decorators.
    """

    classes: list[Class] = []

    for nested in nested_code_objects(
        code
    ):
        if not is_class_body(
            nested
        ):
            continue

        # A class body cannot be distinguished with certainty from
        # an ordinary nested function using only the CodeType object.
        # Avoid treating obvious functions as classes here.
        if nested.co_argcount != 0:
            continue

        if nested.co_name.startswith(
            "<"
        ):
            continue

        classes.append(
            build_class(
                nested
            )
        )

    return classes