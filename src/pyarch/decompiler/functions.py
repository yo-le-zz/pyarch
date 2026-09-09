from __future__ import annotations

"""
Function reconstruction for PyArch.

This module handles Python function code objects and converts their
metadata into the PyArch IR.

It does not yet reconstruct the complete function body. Body
reconstruction is delegated to the statement / engine layers.
"""


from dataclasses import dataclass

from .ir import (
    Function,
    IRExpression,
    IRStatement,
    Name,
)
from ..pyc import iter_code_objects
from types import CodeType


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FunctionError(Exception):
    """Raised when a function cannot be reconstructed."""


# ---------------------------------------------------------------------------
# Parameter information
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ParameterInfo:
    """Information about one Python function parameter."""

    name: str

    kind: str = "positional"

    default_index: int | None = None

    annotation: IRExpression | None = None


# ---------------------------------------------------------------------------
# Code object helpers
# ---------------------------------------------------------------------------


def _parameter_names(
    code: CodeType,
) -> list[str]:
    """
    Return parameter names in Python source order.

    CPython stores positional-only and positional-or-keyword parameters
    together at the beginning of co_varnames.
    """

    positional_count = (
        code.co_posonlyargcount
        + code.co_argcount
    )

    positional = list(
        code.co_varnames[
            :positional_count
        ]
    )

    index = positional_count

    keyword_only = list(
        code.co_varnames[
            index : index
            + code.co_kwonlyargcount
        ]
    )

    index += code.co_kwonlyargcount

    parameters = positional + keyword_only

    if code.co_flags & _VARARGS:
        if index >= len(code.co_varnames):
            raise FunctionError(
                "Function declares *args but "
                "its variable metadata is incomplete."
            )

        parameters.append(
            "*" + code.co_varnames[index]
        )

        index += 1

    elif code.co_kwonlyargcount:
        # If there is no *args parameter, keyword-only parameters
        # still require a bare "*" separator in the source.
        marker = "*"

        if marker not in parameters:
            parameters.insert(
                code.co_posonlyargcount
                + code.co_argcount,
                marker,
            )

    if code.co_flags & _VARKEYWORDS:
        if index >= len(code.co_varnames):
            raise FunctionError(
                "Function declares **kwargs but "
                "its variable metadata is incomplete."
            )

        parameters.append(
            "**" + code.co_varnames[index]
        )

    return parameters


# CPython code-object flags.
_VARARGS = 0x0004
_VARKEYWORDS = 0x0008
_GENERATOR = 0x0020
_COROUTINE = 0x0080
_ASYNC_GENERATOR = 0x0200


def is_generator(
    code: CodeType,
) -> bool:
    """Return whether a code object represents a generator."""

    return bool(
        code.co_flags & _GENERATOR
    )


def is_coroutine(
    code: CodeType,
) -> bool:
    """Return whether a code object represents a coroutine."""

    return bool(
        code.co_flags & _COROUTINE
    )


def is_async_generator(
    code: CodeType,
) -> bool:
    """Return whether a code object represents an async generator."""

    return bool(
        code.co_flags & _ASYNC_GENERATOR
    )


def is_function_code(
    code: CodeType,
) -> bool:
    """
    Determine whether a code object is likely a function body.

    Nested code objects can represent lambdas, comprehensions,
    generators, or other compiler-generated constructs, so this is
    deliberately conservative.
    """

    name = code.co_name

    if name == "<module>":
        return False

    return True


# ---------------------------------------------------------------------------
# Function metadata
# ---------------------------------------------------------------------------


def inspect_parameters(
    code: CodeType,
) -> list[ParameterInfo]:
    """
    Extract parameter metadata from a code object.
    """

    positional_count = (
        code.co_posonlyargcount
        + code.co_argcount
    )

    parameters: list[ParameterInfo] = []

    for index, name in enumerate(
        code.co_varnames[
            :positional_count
        ]
    ):
        if index < code.co_posonlyargcount:
            kind = "positional-only"
        else:
            kind = "positional-or-keyword"

        parameters.append(
            ParameterInfo(
                name=name,
                kind=kind,
            )
        )

    offset = positional_count

    for name in code.co_varnames[
        offset : offset
        + code.co_kwonlyargcount
    ]:
        parameters.append(
            ParameterInfo(
                name=name,
                kind="keyword-only",
            )
        )

    offset += code.co_kwonlyargcount

    if code.co_flags & _VARARGS:
        if offset < len(code.co_varnames):
            parameters.append(
                ParameterInfo(
                    name=code.co_varnames[offset],
                    kind="var-positional",
                )
            )

            offset += 1

    if code.co_flags & _VARKEYWORDS:
        if offset < len(code.co_varnames):
            parameters.append(
                ParameterInfo(
                    name=code.co_varnames[offset],
                    kind="var-keyword",
                )
            )

    return parameters


# ---------------------------------------------------------------------------
# Function construction
# ---------------------------------------------------------------------------


def build_function(
    code: CodeType,
    *,
    body: list[IRStatement] | None = None,
    decorators: list[IRExpression] | None = None,
) -> Function:
    """
    Build a Function IR node from a code object.

    The body can be supplied by the main decompiler engine.
    """

    parameters = inspect_parameters(
        code
    )

    parameter_names: list[str] = []

    for parameter in parameters:
        if parameter.kind == "var-positional":
            parameter_names.append(
                "*" + parameter.name
            )

        elif parameter.kind == "var-keyword":
            parameter_names.append(
                "**" + parameter.name
            )

        else:
            parameter_names.append(
                parameter.name
            )

    # Insert "*" before keyword-only parameters when no *args exists.
    has_varargs = any(
        parameter.kind == "var-positional"
        for parameter in parameters
    )

    has_keyword_only = any(
        parameter.kind == "keyword-only"
        for parameter in parameters
    )

    if has_keyword_only and not has_varargs:
        positional_end = (
            code.co_posonlyargcount
            + code.co_argcount
        )

        parameter_names.insert(
            positional_end,
            "*",
        )

    return Function(
        offset=None,
        name=code.co_name,
        parameters=parameter_names,
        body=[] if body is None else body,
        decorators=(
            []
            if decorators is None
            else decorators
        ),
        is_async=(
            is_coroutine(code)
            or is_async_generator(code)
        ),
        is_generator=(
            is_generator(code)
            or is_async_generator(code)
        ),
    )


# ---------------------------------------------------------------------------
# Nested code objects
# ---------------------------------------------------------------------------


def nested_code_objects(
    code: CodeType,
) -> list[CodeType]:
    """
    Return nested code objects directly contained by `code`.

    Unlike iter_code_objects(), this does not recursively flatten the
    entire tree.
    """

    result: list[CodeType] = []

    for constant in code.co_consts:
        if isinstance(
            constant,
            CodeType,
        ):
            result.append(
                constant
            )

    return result


def discover_functions(
    code: CodeType,
) -> list[Function]:
    """
    Discover functions nested inside a code object.

    Bodies are intentionally empty at this stage.
    """

    functions: list[Function] = []

    for nested in nested_code_objects(
        code
    ):
        if not is_function_code(nested):
            continue

        functions.append(
            build_function(
                nested
            )
        )

    return functions


# ---------------------------------------------------------------------------
# Function name helpers
# ---------------------------------------------------------------------------


def is_lambda(
    code: CodeType,
) -> bool:
    """Return whether the code object represents a lambda."""

    return code.co_name == "<lambda>"


def is_comprehension(
    code: CodeType,
) -> bool:
    """Return whether the code object is compiler-generated for a comprehension."""

    return code.co_name in {
        "<listcomp>",
        "<setcomp>",
        "<dictcomp>",
        "<genexpr>",
    }


def function_display_name(
    code: CodeType,
) -> str:
    """
    Return a useful display name for a code object.
    """

    if is_lambda(code):
        return "<lambda>"

    if is_comprehension(code):
        return code.co_name

    return code.co_name


# ---------------------------------------------------------------------------
# Source rendering
# ---------------------------------------------------------------------------


def render_function_signature(
    function: Function,
) -> str:
    """
    Render a function signature.

    Defaults and annotations will be added by later reconstruction
    passes once MAKE_FUNCTION metadata has been decoded.
    """

    prefix = (
        "async def"
        if function.is_async
        else "def"
    )

    name = function.name

    if name == "<lambda>":
        raise FunctionError(
            "Lambda functions do not use def syntax."
        )

    parameters = ", ".join(
        function.parameters
    )

    return (
        f"{prefix} "
        f"{name}({parameters})"
    )


def render_function(
    function: Function,
    *,
    indent: int = 0,
) -> str:
    """Render a function with its current IR body."""

    prefix = "    " * indent

    signature = (
        prefix
        + render_function_signature(
            function
        )
        + ":"
    )

    if not function.body:
        return (
            signature
            + "\n"
            + prefix
            + "    pass"
        )

    from .statements import render_statements

    body = render_statements(
        function.body,
        indent=indent + 1,
    )

    return (
        signature
        + "\n"
        + body
    )