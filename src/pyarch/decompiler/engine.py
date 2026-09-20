from __future__ import annotations

from .writer import write_and_validate

"""
Main decompiler engine for PyArch.

Pipeline:

    CodeType
       ↓
    translation
       ↓
    CFG
       ↓
    stack / expression reconstruction
       ↓
    statements
       ↓
    functions / classes / comprehensions
       ↓
    IR Module

This layer is deliberately responsible for orchestration rather than
implementing individual bytecode rules. Version-specific bytecode
logic belongs in translate.py / future versions/* modules.
"""

from dataclasses import dataclass, field
from types import CodeType

from ..coderef import is_code_object
from typing import Iterable

from .cfg import build_cfg
from .comprehensions import (
    Comprehension,
    ComprehensionError,
    analyze_comprehension,
    detect_inline_comprehension,
    is_comprehension_code,
)
from .control_flow import build_structured_body, collapse_rotated_while
from .exceptions import (
    ExceptionRegion,
    reconstruct_exceptions,
)
from .functions import (
    Function,
    build_function,
    is_function_code,
    nested_code_objects as nested_function_code_objects,
)
from .ir import (
    Class,
    ControlFlowGraph,
    IRStatement,
    Module,
)
from .loops import (
    Loop,
    reconstruct_loops,
)
from .statements import (
    reconstruct_statements,
)
from .translate import (
    TInstruction,
    translate_code,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DecompilerError(Exception):
    """Raised when bytecode cannot be reconstructed."""


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecompileDiagnostics:
    """
    Information collected during decompilation.

    PyArch should never silently pretend that something was recovered
    when it was actually skipped.
    """

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    skipped: list[str] = field(
        default_factory=list
    )

    recovered: list[str] = field(
        default_factory=list
    )

    # Transient scratch space used internally by control_flow.py to
    # thread try/except context through the recursive structuring
    # pass without adding a new parameter to every function in the
    # call chain. Not diagnostic output; never read by callers.
    _pyarch_by_offset: dict | None = None
    _pyarch_sorted_offsets: list | None = None
    _pyarch_try_starts: dict | None = None

    def warn(
        self,
        message: str,
    ) -> None:
        self.warnings.append(message)

    def error(
        self,
        message: str,
    ) -> None:
        self.errors.append(message)

    def skip(
        self,
        message: str,
    ) -> None:
        self.skipped.append(message)

    def recover(
        self,
        message: str,
    ) -> None:
        self.recovered.append(message)


@dataclass(slots=True)
class DecompiledCode:
    """
    Result of decompiling one CodeType.
    """

    code: CodeType

    instructions: list[TInstruction]

    cfg: ControlFlowGraph | None = None

    statements: list[IRStatement] = field(
        default_factory=list
    )

    loops: list[Loop] = field(
        default_factory=list
    )

    exceptions: list[ExceptionRegion] = field(
        default_factory=list
    )

    functions: list[Function] = field(
        default_factory=list
    )

    classes: list[Class] = field(
        default_factory=list
    )

    comprehensions: list[Comprehension] = field(
        default_factory=list
    )


@dataclass(slots=True)
class DecompileResult:
    """
    Complete result of decompiling a module.
    """

    module: Module

    diagnostics: DecompileDiagnostics

    code: DecompiledCode

    source: str = ""


# ---------------------------------------------------------------------------
# Code-object discovery
# ---------------------------------------------------------------------------


def nested_code_objects(
    code: CodeType,
) -> list[CodeType]:
    """Return direct child code objects."""

    return [
        constant
        for constant in code.co_consts
        if is_code_object(constant)
    ]


def iter_code_objects(
    code: CodeType,
):
    """Recursively yield all code objects."""

    yield code

    for nested in nested_code_objects(
        code
    ):
        yield from iter_code_objects(
            nested
        )


# ---------------------------------------------------------------------------
# Code classification
# ---------------------------------------------------------------------------


_GENERATED_NAMES = {
    "<module>",
    "<lambda>",
    "<listcomp>",
    "<setcomp>",
    "<dictcomp>",
    "<genexpr>",
    "<async_generator>",
}


def is_generated_code(
    code: CodeType,
) -> bool:
    """Return whether a code object has a compiler-generated name."""

    return (
        code.co_name in _GENERATED_NAMES
        or code.co_name.startswith("<")
    )


def is_lambda_code(
    code: CodeType,
) -> bool:
    return code.co_name == "<lambda>"


def is_comprehension(
    code: CodeType,
) -> bool:
    return is_comprehension_code(
        code
    )


def looks_like_class_code(
    code: CodeType,
) -> bool:
    """
    Detect a class-body code object.

    CPython always compiles a class body to store ``__module__`` and
    ``__qualname__`` via STORE_NAME at the very start, so both names
    are guaranteed to appear in ``co_names``. A plain function never
    stores those names, regardless of its argument count -- which is
    why a heuristic based on argument count alone (a zero-argument
    nested function looks identical to a class body under that
    heuristic) is not reliable.
    """

    if is_generated_code(code):
        return False

    return (
        "__module__" in code.co_names
        and "__qualname__" in code.co_names
    )


# ---------------------------------------------------------------------------
# Instruction / CFG stage
# ---------------------------------------------------------------------------


def translate(
    code: CodeType,
) -> list[TInstruction]:
    """Translate raw CPython instructions to PyArch instructions."""

    try:
        return translate_code(
            code
        )
    except Exception as error:
        raise DecompilerError(
            f"Could not translate bytecode "
            f"for {code.co_name!r}."
        ) from error


def build_control_flow(
    instructions: list[TInstruction],
) -> ControlFlowGraph:
    """Build a control-flow graph from translated instructions."""

    try:
        return build_cfg(
            instructions
        )
    except Exception as error:
        raise DecompilerError(
            "Could not build control-flow graph."
        ) from error


# ---------------------------------------------------------------------------
# Statement stage
# ---------------------------------------------------------------------------


def reconstruct_code_statements(
    instructions: list[TInstruction],
    diagnostics: DecompileDiagnostics,
) -> list[IRStatement]:
    """
    Reconstruct straight-line statements.

    ``reconstruct_statements()`` returns a ``StatementResult`` rather
    than a raw list, so this function unwraps it here and keeps the
    rest of the engine independent from that implementation detail.
    """

    try:
        result = reconstruct_statements(
            instructions
        )
    except Exception as error:
        diagnostics.error(
            f"Statement reconstruction failed: {error}"
        )
        return []

    # StatementResult is the public result object returned by
    # statements.py.  The actual statements are stored in ``statements``.
    statements = result.statements

    diagnostics.recover(
        f"Recovered {len(statements)} statement(s)."
    )

    return statements


# ---------------------------------------------------------------------------
# Structural stages
# ---------------------------------------------------------------------------


def reconstruct_code_loops(
    cfg: ControlFlowGraph,
    diagnostics: DecompileDiagnostics,
) -> list[Loop]:
    """Recover loops from the CFG."""

    try:
        loops = reconstruct_loops(
            cfg
        )
    except Exception as error:
        diagnostics.warn(
            f"Loop reconstruction failed: {error}"
        )
        return []

    if loops:
        diagnostics.recover(
            f"Recovered {len(loops)} loop(s)."
        )

    return loops


def reconstruct_code_exceptions(
    cfg: ControlFlowGraph,
    instructions: list[TInstruction],
    diagnostics: DecompileDiagnostics,
) -> list[ExceptionRegion]:
    """Recover exception regions conservatively."""

    try:
        regions = reconstruct_exceptions(
            cfg,
        )
    except Exception as error:
        diagnostics.warn(
            f"Exception reconstruction failed: {error}"
        )
        return []

    if regions:
        diagnostics.recover(
            f"Recovered {len(regions)} exception region(s)."
        )

    return regions


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


def discover_functions(
    code: CodeType,
) -> list[Function]:
    """Discover ordinary nested functions."""

    functions: list[Function] = []

    for nested in nested_code_objects(
        code
    ):
        if is_comprehension(nested):
            continue

        if is_lambda_code(nested):
            continue

        if looks_like_class_code(nested):
            continue

        try:
            if is_function_code(nested):
                functions.append(
                    build_function(
                        nested
                    )
                )
        except Exception:
            continue

    return functions


def discover_classes(
    code: CodeType,
) -> list[Class]:
    """Discover likely nested class bodies."""

    from .classes import (
        build_class,
    )

    classes: list[Class] = []

    for nested in nested_code_objects(
        code
    ):
        if not looks_like_class_code(
            nested
        ):
            continue

        try:
            classes.append(
                build_class(
                    nested
                )
            )
        except Exception:
            continue

    return classes


def discover_comprehension_objects(
    code: CodeType,
    diagnostics: DecompileDiagnostics,
) -> list[Comprehension]:
    """Discover traditional nested comprehensions."""

    result: list[Comprehension] = []

    for nested in nested_code_objects(
        code
    ):
        if not is_comprehension(
            nested
        ):
            continue

        try:
            result.append(
                analyze_comprehension(
                    nested
                )
            )
        except ComprehensionError as error:
            diagnostics.warn(
                f"Could not analyze comprehension "
                f"{nested.co_name!r}: {error}"
            )

    return result


def discover_inline_comprehension(
    code: CodeType,
    diagnostics: DecompileDiagnostics,
) -> list[Comprehension]:
    """Detect Python 3.13-style inline comprehensions."""

    try:
        structure = detect_inline_comprehension(
            code
        )
    except Exception as error:
        diagnostics.warn(
            f"Inline comprehension detection failed: "
            f"{error}"
        )
        return []

    if structure is None:
        return []

    return [
        Comprehension(
            kind=structure.kind,
            source=code,
            inline=True,
        )
    ]


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------


def _strip_trailing_none_returns(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Recursively drop a trailing ``return None`` wherever it is the
    *last* statement of its own list (module body, function body, or
    any nested if/while/for/try branch).

    Only ever removing a genuinely trailing one keeps this safe for
    function bodies too: nothing after it in the same list could
    become newly reachable, since there was nothing after it to begin
    with. This is what turns CPython's tail-position duplication of
    an implicit return (into every branch of a final if/try) back
    into the plain, return-free branches the source almost certainly
    had.
    """

    from .ir import Return, Constant, If, While, For, Try

    result = list(statements)

    for statement in result:
        if isinstance(statement, (While, For)):
            statement.body = _strip_trailing_none_returns(
                statement.body
            )

    for statement in result:
        if isinstance(statement, If):
            statement.body = _strip_trailing_none_returns(
                statement.body
            )
            statement.orelse = _strip_trailing_none_returns(
                statement.orelse
            )
        elif isinstance(statement, Try):
            statement.body = _strip_trailing_none_returns(
                statement.body
            )
            for handler in statement.handlers:
                handler.body = _strip_trailing_none_returns(
                    handler.body
                )
            statement.orelse = _strip_trailing_none_returns(
                statement.orelse
            )
            statement.finalbody = _strip_trailing_none_returns(
                statement.finalbody
            )

    if result:
        last = result[-1]

        if isinstance(last, Return):
            value = getattr(last, "value", None)

            if value is None or (
                isinstance(value, Constant)
                and value.value is None
            ):
                result = result[:-1]

    return result


def _strip_redundant_trailing_continue(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Simplify ``if cond: body; continue else: continue`` back to
    ``if cond: body`` when it is the last statement of a loop body:
    falling off the end of a loop body and an explicit `continue`
    there are behaviorally identical, so when *every* branch of a
    trailing if is just "do something, then continue" (or nothing,
    then continue), the continues carry no information and the
    reconstruction that produced them was really just an ordinary
    `if` with no `else`.
    """

    from .ir import If, While, For, Continue, Pass

    result = list(statements)

    for statement in result:
        if isinstance(statement, (While, For)):
            statement.body = _strip_redundant_trailing_continue(
                statement.body
            )

    for statement in result:
        if isinstance(statement, If):
            statement.body = _strip_redundant_trailing_continue(
                statement.body
            )
            statement.orelse = _strip_redundant_trailing_continue(
                statement.orelse
            )

    if result and isinstance(result[-1], If):
        last_if = result[-1]

        def _is_bare_continue_tail(
            body: list[IRStatement],
        ) -> bool:
            return bool(body) and isinstance(
                body[-1], Continue
            )

        if _is_bare_continue_tail(
            last_if.body
        ) and _is_bare_continue_tail(last_if.orelse):
            # Both branches: ...; continue -- the continues are
            # exactly what falling off the end of the loop body
            # would do anyway, so they carry no information.
            new_orelse = last_if.orelse[:-1]

            last_if.body = last_if.body[:-1] or [Pass()]
            last_if.orelse = new_orelse

            result[-1] = last_if

    return result


def _remove_module_return(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Remove the implicit trailing ``return None``.

    CPython terminates every code object's bytecode with a return
    instruction, but that is not source-level Python and must not
    appear in the reconstructed module/function/class body.
    """

    if not statements:
        return statements

    last = statements[-1]

    from .ir import Return, Constant

    if isinstance(last, Return):
        value = getattr(
            last,
            "value",
            None,
        )

        if isinstance(
            value,
            Constant,
        ) and value.value is None:
            return statements[:-1]

    return statements


def _merge_consecutive_imports(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    CPython emits one IMPORT_FROM per imported name, which the
    statement layer turns into one `ImportFrom` each. Merge
    consecutive ones from the same module/level back into a single
    `from module import a, b, c` for readability.
    """

    from .ir import ImportFrom, If, While, For

    result: list[IRStatement] = []

    for statement in statements:
        if isinstance(statement, If):
            statement.body = _merge_consecutive_imports(
                statement.body
            )
            statement.orelse = _merge_consecutive_imports(
                statement.orelse
            )
        elif isinstance(statement, (While, For)):
            statement.body = _merge_consecutive_imports(
                statement.body
            )

        if (
            result
            and isinstance(statement, ImportFrom)
            and isinstance(result[-1], ImportFrom)
            and result[-1].module == statement.module
            and result[-1].level == statement.level
        ):
            result[-1].names.extend(statement.names)
            continue

        result.append(statement)

    return result


_CLASS_BOILERPLATE_NAMES = {
    "__qualname__",
    "__module__",
    "__firstlineno__",
}

_CLASS_TRAILING_BOILERPLATE_NAMES = {
    "__static_attributes__",
    "__classcell__",
}


def _strip_class_boilerplate(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Drop the compiler-generated bookkeeping assignments that begin
    and end every class body (``__qualname__``, ``__module__``,
    ``__firstlineno__`` at the start; ``__static_attributes__`` and
    ``__classcell__`` at the end, along with the trailing
    ``return __class__`` classes using ``super()``/``__class__``
    compile in). None of these carry source-level meaning.
    """

    from .ir import Assign, Name, Return

    result = list(statements)

    while result:
        first = result[0]

        if (
            isinstance(first, Assign)
            and isinstance(first.target, Name)
            and first.target.name in _CLASS_BOILERPLATE_NAMES
        ):
            result.pop(0)
            continue

        break

    while result:
        last = result[-1]

        if isinstance(last, Return):
            value = last.value

            if isinstance(value, Name) and value.name == (
                "__class__"
            ):
                result.pop()
                continue

        if (
            isinstance(last, Assign)
            and isinstance(last.target, Name)
            and last.target.name
            in _CLASS_TRAILING_BOILERPLATE_NAMES
        ):
            result.pop()
            continue

        break

    return result


def _strip_module_returns(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Module-level code cannot contain a real ``return`` statement, yet
    when an if/while/for is the final top-level statement, CPython
    often has *each* branch end with its own ``RETURN_CONST None``
    instead of jumping to a shared exit. Recursively drop every such
    implicit-None return, in any nested body, so the generated source
    stays valid at module scope.
    """

    from .ir import Return, Constant, If, While, For, Try

    result: list[IRStatement] = []

    for statement in statements:
        if isinstance(statement, Return):
            value = getattr(statement, "value", None)

            if value is None or (
                isinstance(value, Constant)
                and value.value is None
            ):
                continue

        if isinstance(statement, If):
            statement.body = _strip_module_returns(
                statement.body
            )
            statement.orelse = _strip_module_returns(
                statement.orelse
            )
        elif isinstance(statement, (While, For)):
            statement.body = _strip_module_returns(
                statement.body
            )
        elif isinstance(statement, Try):
            statement.body = _strip_module_returns(
                statement.body
            )
            for handler in statement.handlers:
                handler.body = _strip_module_returns(
                    handler.body
                )
            statement.orelse = _strip_module_returns(
                statement.orelse
            )
            statement.finalbody = _strip_module_returns(
                statement.finalbody
            )

        result.append(statement)

    return result


def build_module(
    code: CodeType,
    statements: list[IRStatement],
) -> Module:
    """Build the top-level PyArch Module IR."""

    return Module(
        offset=None,
        name=code.co_name,
        body=_remove_module_return(
            statements
        ),
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class DecompilerEngine:
    """
    Main bytecode-to-IR reconstruction engine.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
    ) -> None:
        self.strict = strict

    def decompile(
        self,
        code: CodeType,
    ) -> DecompileResult:
        """
        Decompile one top-level CodeType.
        """

        diagnostics = DecompileDiagnostics()

        # ---------------------------------------------------------------
        # 1. Translation
        # ---------------------------------------------------------------

        instructions = translate(
            code
        )

        diagnostics.recover(
            f"Translated {len(instructions)} instruction(s)."
        )

        from .comprehension_rewrite import rewrite_comprehensions

        instructions = rewrite_comprehensions(
            instructions, diagnostics
        )

        from .boolop_rewrite import rewrite_boolops

        instructions = rewrite_boolops(
            instructions, diagnostics
        )

        # ---------------------------------------------------------------
        # 2. CFG
        # ---------------------------------------------------------------

        try:
            cfg = build_control_flow(
                instructions
            )
        except DecompilerError as error:
            diagnostics.error(
                str(error)
            )

            if self.strict:
                raise

            cfg = None

        # ---------------------------------------------------------------
        # 3. Nested functions/classes (recursive, body-first)
        # ---------------------------------------------------------------
        #
        # A nested `def`/`class` can only be inlined correctly once we
        # know its own body, so every nested code object is fully
        # decompiled *before* this code object's own statements are
        # structured.

        code_map: dict[int, Function | Class] = {}

        for nested in nested_code_objects(code):
            if is_comprehension(nested) or is_lambda_code(
                nested
            ):
                # Comprehensions/lambdas are handled inline by the
                # comprehension/expression layers, not as def/class
                # statements.
                continue

            try:
                nested_result = self.decompile(nested)
            except Exception as error:
                diagnostics.warn(
                    f"Could not decompile nested code object "
                    f"{nested.co_name!r}: {error}"
                )
                continue

            diagnostics.warnings.extend(
                nested_result.diagnostics.warnings
            )
            diagnostics.errors.extend(
                nested_result.diagnostics.errors
            )

            body = nested_result.module.body

            if looks_like_class_code(nested):
                from .classes import build_class

                try:
                    code_map[id(nested)] = build_class(
                        nested,
                        body=_strip_class_boilerplate(
                            body
                        ),
                    )
                except Exception as error:
                    diagnostics.warn(
                        f"Could not build class "
                        f"{nested.co_name!r}: {error}"
                    )
            elif is_function_code(nested):
                try:
                    code_map[id(nested)] = build_function(
                        nested,
                        body=body,
                    )
                except Exception as error:
                    diagnostics.warn(
                        f"Could not build function "
                        f"{nested.co_name!r}: {error}"
                    )

        # ---------------------------------------------------------------
        # 4. Structured statements (if/while/for, with inline def/class)
        # ---------------------------------------------------------------

        statements = build_structured_body(
            cfg,
            instructions,
            code_map,
            diagnostics,
            code=code,
        )

        statements = _remove_module_return(
            statements
        )

        if code.co_name == "<module>":
            statements = _strip_module_returns(
                statements
            )

        statements = _strip_trailing_none_returns(
            statements
        )

        statements = _strip_redundant_trailing_continue(
            statements
        )

        statements = collapse_rotated_while(
            statements
        )

        statements = _merge_consecutive_imports(
            statements
        )

        diagnostics.recover(
            f"Recovered {len(statements)} statement(s)."
        )

        # ---------------------------------------------------------------
        # 5. Structural analysis (informational; structuring above
        #    already reconstructs loops/exceptions where possible)
        # ---------------------------------------------------------------

        loops: list[Loop] = []
        exceptions: list[ExceptionRegion] = []
        # Note: the legacy `reconstruct_code_exceptions` heuristic
        # (CFG-block-based) is superseded by the exception-table-based
        # `Try`/`ExceptHandler` reconstruction in control_flow.py,
        # which is what actually drives the generated source. This
        # field is kept for API compatibility but no longer computed
        # here to avoid a confusing, unrelated internal warning.

        # ---------------------------------------------------------------
        # 6. Nested functions/classes/comprehensions (metadata/counts)
        # ---------------------------------------------------------------

        functions = [
            value
            for value in code_map.values()
            if isinstance(value, Function)
        ]

        classes = [
            value
            for value in code_map.values()
            if isinstance(value, Class)
        ]

        comprehensions = (
            discover_comprehension_objects(
                code,
                diagnostics,
            )
        )

        # Python 3.13 can encode comprehensions directly in the
        # surrounding code object.
        if not comprehensions:
            comprehensions.extend(
                discover_inline_comprehension(
                    code,
                    diagnostics,
                )
            )

        # ---------------------------------------------------------------
        # 7. Module
        # ---------------------------------------------------------------

        module = build_module(
            code,
            statements,
        )

        result_code = DecompiledCode(
            code=code,
            instructions=instructions,
            cfg=cfg,
            statements=statements,
            loops=loops,
            exceptions=exceptions,
            functions=functions,
            classes=classes,
            comprehensions=comprehensions,
        )

        source = ""

        try:
            source = write_and_validate(module)
            diagnostics.recover(
                f"Generated {len(source.splitlines())} source line(s)."
            )
        except Exception as error:
            diagnostics.warn(
                f"Could not generate Python source: {error}"
            )

        return DecompileResult(
            module=module,
            diagnostics=diagnostics,
            code=result_code,
            source=source,
        )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def decompile_code(
    code: CodeType,
    *,
    strict: bool = False,
) -> DecompileResult:
    """Convenience wrapper around :class:`DecompilerEngine`."""

    return DecompilerEngine(
        strict=strict
    ).decompile(
        code
    )


__all__ = [
    "DecompilerError",
    "DecompileDiagnostics",
    "DecompiledCode",
    "DecompileResult",
    "DecompilerEngine",
    "decompile_code",
    "nested_code_objects",
    "iter_code_objects",
    "is_generated_code",
    "is_lambda_code",
    "is_comprehension",
    "looks_like_class_code",
    "translate",
    "build_control_flow",
    "reconstruct_code_statements",
    "reconstruct_code_loops",
    "reconstruct_code_exceptions",
    "discover_functions",
    "discover_classes",
    "discover_comprehension_objects",
    "discover_inline_comprehension",
    "build_module",
]