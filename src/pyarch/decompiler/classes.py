from __future__ import annotations

"""
Class reconstruction for PyArch.

This module detects and describes Python class code objects.

Class construction is deliberately conservative. The surrounding
bytecode is required to recover bases, decorators and metaclass
arguments reliably; those details will be handled by the main
decompiler engine.
"""

from dataclasses import dataclass
from types import CodeType

from .ir import (
    Class,
    Function,
    IRExpression,
    IRStatement,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClassError(Exception):
    """Raised when class reconstruction fails."""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassMetadata:
    """Metadata recovered from a Python class body."""

    name: str
    bases: list[IRExpression]
    decorators: list[IRExpression]
    methods: list[Function]
    body: list[IRStatement]


# ---------------------------------------------------------------------------
# Code object helpers
# ---------------------------------------------------------------------------


def nested_code_objects(
    code: CodeType,
) -> list[CodeType]:
    """Return code objects directly contained in a code object."""

    result: list[CodeType] = []

    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            result.append(constant)

    return result


def is_class_code(
    code: CodeType,
) -> bool:
    """
    Return whether a code object could represent a class body.

    This is only a heuristic. A CodeType does not contain an explicit
    "this is a class" flag.
    """

    if code.co_name == "<module>":
        return False

    if code.co_name.startswith("<"):
        return False

    # Class bodies normally do not receive normal Python arguments.
    if code.co_argcount != 0:
        return False

    if code.co_kwonlyargcount != 0:
        return False

    if code.co_posonlyargcount != 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Method detection
# ---------------------------------------------------------------------------


_GENERATED_CODE_NAMES = {
    "<lambda>",
    "<listcomp>",
    "<setcomp>",
    "<dictcomp>",
    "<genexpr>",
    "<async_generator>",
}


def is_method_code(
    code: CodeType,
) -> bool:
    """
    Return whether a nested code object looks like a method.

    Compiler-generated helper code is excluded.
    """

    if code.co_name in _GENERATED_CODE_NAMES:
        return False

    if code.co_name == "<module>":
        return False

    return True


def discover_methods(
    code: CodeType,
) -> list[Function]:
    """
    Discover method code objects directly contained in a class body.

    Function bodies are reconstructed later by the main engine.
    """

    from .functions import build_function

    methods: list[Function] = []

    for nested in nested_code_objects(code):
        if not is_method_code(nested):
            continue

        methods.append(
            build_function(nested)
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


def find_method(
    class_ir: Class,
    name: str,
) -> Function | None:
    """
    Find a method in a class.

    The current IR stores methods separately only in later engine
    stages, so this helper is intentionally conservative.
    """

    methods = getattr(
        class_ir,
        "methods",
        None,
    )

    if methods is None:
        return None

    for method in methods:
        if method.name == name:
            return method

    return None


# ---------------------------------------------------------------------------
# Class construction
# ---------------------------------------------------------------------------


def build_class(
    code: CodeType,
    *,
    bases: list[IRExpression] | None = None,
    decorators: list[IRExpression] | None = None,
    body: list[IRStatement] | None = None,
) -> Class:
    """
    Build a Class IR node from a class-body code object.

    Bases and decorators are supplied separately because they are
    encoded in the surrounding bytecode rather than the class
    CodeType itself.
    """

    return Class(
        offset=None,
        name=code.co_name,
        bases=(
            []
            if bases is None
            else bases
        ),
        body=(
            []
            if body is None
            else body
        ),
        decorators=(
            []
            if decorators is None
            else decorators
        ),
    )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_class(
    code: CodeType,
) -> ClassMetadata:
    """
    Inspect a class-body code object.

    This does not attempt to reconstruct the complete class yet.
    """

    if not is_class_code(code):
        raise ClassError(
            f"Code object {code.co_name!r} "
            "does not look like a class body."
        )

    methods = discover_methods(
        code
    )

    return ClassMetadata(
        name=code.co_name,
        bases=[],
        decorators=[],
        methods=methods,
        body=[],
    )


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def class_attribute_names(
    code: CodeType,
) -> list[str]:
    """
    Return names referenced by the class body.

    This is not yet an exact list of assignments. The engine will
    distinguish LOAD_NAME / STORE_NAME operations later.
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
# Discovery
# ---------------------------------------------------------------------------


def discover_classes(
    code: CodeType,
) -> list[Class]:
    """
    Discover class-like code objects directly contained in `code`.

    Because CodeType does not explicitly identify class bodies, this
    function uses conservative heuristics.
    """

    classes: list[Class] = []

    for nested in nested_code_objects(code):
        if not is_class_code(nested):
            continue

        classes.append(
            build_class(nested)
        )

    return classes


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_expression(
    expression: IRExpression,
) -> str:
    """Render an IR expression."""

    from .expressions import render_expression

    return render_expression(
        expression
    ).text


def render_class_signature(
    class_ir: Class,
) -> str:
    """Render the class declaration."""

    bases = getattr(
        class_ir,
        "bases",
        [],
    )

    if bases:
        rendered_bases = ", ".join(
            _render_expression(base)
            for base in bases
        )

        return (
            f"class {class_ir.name}"
            f"({rendered_bases})"
        )

    return f"class {class_ir.name}"


def render_class(
    class_ir: Class,
    *,
    indent: int = 0,
) -> str:
    """
    Render a class for debugging.

    Final Python generation will eventually use Python's AST module.
    """

    prefix = "    " * indent

    lines = [
        prefix
        + render_class_signature(
            class_ir
        )
        + ":"
    ]

    body = getattr(
        class_ir,
        "body",
        [],
    )

    if body:
        from .statements import render_statements

        lines.append(
            render_statements(
                body,
                indent=indent + 1,
            )
        )

    methods = getattr(
        class_ir,
        "methods",
        [],
    )

    if methods:
        from .functions import render_function

        for method in methods:
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