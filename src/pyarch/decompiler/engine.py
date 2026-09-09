from __future__ import annotations

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
from typing import Iterable

from .cfg import build_cfg
from .comprehensions import (
    Comprehension,
    ComprehensionError,
    analyze_comprehension,
    detect_inline_comprehension,
    is_comprehension_code,
)
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
        if isinstance(
            constant,
            CodeType,
        )
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
    Heuristic class-body detection.

    A class body normally has no normal function arguments and has
    a regular identifier as its code name.
    """

    if is_generated_code(code):
        return False

    if code.co_argcount != 0:
        return False

    if code.co_posonlyargcount != 0:
        return False

    if code.co_kwonlyargcount != 0:
        return False

    return code.co_name.isidentifier()


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
            instructions,
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


def _remove_module_return(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    Remove the implicit module-level ``return None``.

    CPython terminates module bytecode with a return instruction, but
    that is not source-level Python and must not appear in the
    reconstructed module.
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
        # 3. Statements
        # ---------------------------------------------------------------

        statements = (
            reconstruct_code_statements(
                instructions,
                diagnostics,
            )
        )

        # ---------------------------------------------------------------
        # 4. Structural analysis
        # ---------------------------------------------------------------

        loops: list[Loop] = []
        exceptions: list[ExceptionRegion] = []

        if cfg is not None:
            loops = reconstruct_code_loops(
                cfg,
                diagnostics,
            )

            exceptions = reconstruct_code_exceptions(
                cfg,
                instructions,
                diagnostics,
            )

        # ---------------------------------------------------------------
        # 5. Nested functions/classes
        # ---------------------------------------------------------------

        functions = discover_functions(
            code
        )

        classes = discover_classes(
            code
        )

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
        # 6. Module
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

        return DecompileResult(
            module=module,
            diagnostics=diagnostics,
            code=result_code,
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