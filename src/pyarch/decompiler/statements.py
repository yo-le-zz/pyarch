from __future__ import annotations

"""
Python statement reconstruction.

This module converts semantic bytecode operations and the virtual
evaluation stack into PyArch IR statements.

It intentionally does not try to reconstruct structured control flow
(if/while/for/try). That belongs to later passes.
"""


from dataclasses import dataclass, field, replace
from types import CodeType

from .expressions import render_expression
from .ir import (
    Assign,
    Call,
    Class,
    Constant,
    Delete,
    ExpressionStatement,
    Function,
    Import,
    ImportFrom,
    IRExpression,
    IRStatement,
    Name,
    Pass,
    Return,
    Starred,
    TupleExpr,
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
# Internal markers (never reach the writer)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FunctionRef(IRExpression):
    """
    Transient stack value produced by MAKE_FUNCTION.

    Resolved into a real `Function`/`Class` IR node as soon as it is
    consumed by a STORE instruction. Never reaches the AST writer.
    """

    code: CodeType | None = None
    defaults: list | None = None
    kwdefaults: dict | None = None
    annotations: dict | None = None


@dataclass(slots=True)
class _UnpackSlot(IRExpression):
    """
    Transient stack value produced by UNPACK_SEQUENCE/UNPACK_EX.

    All slots from the same unpack share the same `targets` list;
    each consuming STORE records its target at `position`. Only the
    STORE for the last position (the first one CPython pushed, since
    unpacking pushes right-to-left) turns the whole group into one
    real `Assign` with a tuple target -- earlier ones return nothing,
    so `a, b = seq` reconstructs as one statement instead of two
    invalid ones.
    """

    seq_expr: IRExpression | None = None
    targets: list | None = None
    position: int = 0
    is_starred: bool = False


@dataclass(slots=True)
class _ImportRef(IRExpression):
    """Transient stack value produced by IMPORT_NAME."""

    module: str = ""
    level: int = 0


@dataclass(slots=True)
class _ImportFromRef(IRExpression):
    """Transient stack value produced by IMPORT_FROM."""

    module: str = ""
    name: str = ""
    level: int = 0


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
    code_map: dict[int, "Function | Class"] | None = None,
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
        "MAKE_CELL",
        "COPY_FREE_VARS",
        "LOAD_FAST_AND_CLEAR",
        "PUSH_EXC_INFO",
        "POP_EXCEPT",
    }:
        return None

    # ------------------------------------------------------------------
    # Function / class object construction
    # ------------------------------------------------------------------

    if op == "MAKE_FUNCTION":
        from ..coderef import is_code_object

        try:
            top = stack.pop_expression()
        except StackError as error:
            raise StatementError(
                f"Could not process MAKE_FUNCTION: {error}"
            ) from error

        code_obj: CodeType | None = None

        if isinstance(top, Constant) and is_code_object(
            top.value
        ):
            code_obj = top.value
        else:
            # Pre-3.11 bytecode pushes the qualname above the code
            # object; the value just popped was the qualname.
            try:
                maybe_code = stack.pop_expression()
            except StackError:
                maybe_code = None

            if isinstance(
                maybe_code, Constant
            ) and is_code_object(maybe_code.value):
                code_obj = maybe_code.value

        if code_obj is None:
            raise StatementError(
                "MAKE_FUNCTION: could not locate the code constant."
            )

        # The `arg` is a bitfield: 0x01 defaults, 0x02 kwdefaults,
        # 0x04 annotations, 0x08 closure. Each set bit means one more
        # value sits on the stack below the code object and must be
        # consumed here, or it corrupts every later stack read.
        # Positional defaults are captured on a best-effort basis;
        # the rest are discarded (not yet modelled).
        flags = instruction.arg or 0
        defaults: list[IRExpression] | None = None

        if flags & 0x01:
            try:
                defaults_expr = stack.pop_expression()
            except StackError as error:
                raise StatementError(
                    f"MAKE_FUNCTION: {error}"
                ) from error

            from .ir import TupleExpr

            if isinstance(defaults_expr, TupleExpr):
                defaults = list(defaults_expr.elements)
            elif isinstance(
                defaults_expr, Constant
            ) and isinstance(defaults_expr.value, tuple):
                defaults = [
                    Constant(value=item)
                    for item in defaults_expr.value
                ]

        for bit in (0x02, 0x04, 0x08):
            if flags & bit:
                try:
                    stack.pop_expression()
                except StackError as error:
                    raise StatementError(
                        f"MAKE_FUNCTION: {error}"
                    ) from error

        stack.push(
            _FunctionRef(code=code_obj, defaults=defaults)
        )
        return None

    if op == "SET_FUNCTION_ATTRIBUTE":
        # Python 3.13+: MAKE_FUNCTION always pops just the code
        # object; defaults/kwdefaults/annotations/closure are set
        # afterward by one SET_FUNCTION_ATTRIBUTE per attribute,
        # using the same bit meanings MAKE_FUNCTION's old `arg` had
        # (1 defaults, 2 kwdefaults, 4 annotations, 8 closure).
        try:
            function_ref = stack.pop_expression()
            value = stack.pop_expression()
        except StackError as error:
            raise StatementError(
                f"Could not process SET_FUNCTION_ATTRIBUTE: {error}"
            ) from error

        if not isinstance(function_ref, _FunctionRef):
            raise StatementError(
                "SET_FUNCTION_ATTRIBUTE did not follow a "
                "MAKE_FUNCTION."
            )

        bit = instruction.arg or 0

        if bit == 0x01:
            from .ir import TupleExpr

            if isinstance(value, TupleExpr):
                function_ref.defaults = list(value.elements)
            elif isinstance(value, Constant) and isinstance(
                value.value, tuple
            ):
                function_ref.defaults = [
                    Constant(value=item) for item in value.value
                ]
        elif bit == 0x02:
            function_ref.kwdefaults = _dict_expr_to_mapping(value)
        elif bit == 0x04:
            function_ref.annotations = (
                _flat_annotation_tuple_to_mapping(value)
            )
        # bit 0x08 (closure): consumed above, not modelled further --
        # LOAD_DEREF/STORE_DEREF already reconstruct closure variable
        # access without needing the explicit cell tuple.

        stack.push(function_ref)
        return None

    if op == "UNPACK_SEQUENCE":
        if not isinstance(instruction.arg, int):
            raise StatementError(
                "UNPACK_SEQUENCE has no count."
            )

        count = instruction.arg

        try:
            seq_expr = stack.pop_expression()
        except StackError as error:
            raise StatementError(
                f"Could not process UNPACK_SEQUENCE: {error}"
            ) from error

        targets: list[IRExpression | None] = [None] * count

        # Pushed so that popping (i.e. the STORE order that follows)
        # yields position 0 first, matching left-to-right assignment.
        for position in range(count - 1, -1, -1):
            stack.push(
                _UnpackSlot(
                    seq_expr=seq_expr,
                    targets=targets,
                    position=position,
                )
            )

        return None

    if op == "UNPACK_EX":
        if not isinstance(instruction.arg, int):
            raise StatementError(
                "UNPACK_EX has no counts."
            )

        before = instruction.arg & 0xFF
        after = (instruction.arg >> 8) & 0xFF
        count = before + after + 1

        try:
            seq_expr = stack.pop_expression()
        except StackError as error:
            raise StatementError(
                f"Could not process UNPACK_EX: {error}"
            ) from error

        targets = [None] * count
        star_position = before

        for position in range(count - 1, -1, -1):
            stack.push(
                _UnpackSlot(
                    seq_expr=seq_expr,
                    targets=targets,
                    position=position,
                    is_starred=(position == star_position),
                )
            )

        return None

    if op == "LOAD_BUILD_CLASS":
        stack.push(
            Name(name="__build_class__")
        )
        return None

    if op == "IMPORT_NAME":
        try:
            fromlist_expr = stack.pop_expression()
            level_expr = stack.pop_expression()
        except StackError as error:
            raise StatementError(
                f"Could not process IMPORT_NAME: {error}"
            ) from error

        level = (
            level_expr.value
            if isinstance(level_expr, Constant)
            and isinstance(level_expr.value, int)
            else 0
        )

        module_name = instruction.argval

        if not isinstance(module_name, str):
            raise StatementError(
                "IMPORT_NAME has no valid module name."
            )

        stack.push(
            _ImportRef(module=module_name, level=level)
        )
        return None

    if op == "IMPORT_FROM":
        try:
            top = stack.peek()
        except StackError as error:
            raise StatementError(
                f"Could not process IMPORT_FROM: {error}"
            ) from error

        module_ref = top.expression

        if not isinstance(module_ref, _ImportRef):
            raise StatementError(
                "IMPORT_FROM did not follow an IMPORT_NAME."
            )

        name = instruction.argval

        if not isinstance(name, str):
            raise StatementError(
                "IMPORT_FROM has no valid attribute name."
            )

        stack.push(
            _ImportFromRef(
                module=module_ref.module,
                name=name,
                level=module_ref.level,
            )
        )
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
        "LOAD_CLOSURE",
        "LOAD_FAST_CHECK",
        "LOAD_FAST_LOAD_FAST",
        "PUSH_NULL",
        "LOAD_ATTR",
        "LOAD_SUPER_ATTR",
        "BINARY_OP",
        "UNARY_NOT",
        "UNARY_NEGATIVE",
        "UNARY_POSITIVE",
        "UNARY_INVERT",
        "COMPARE_OP",
        "IS_OP",
        "CONTAINS_OP",
        "CALL",
        "CALL_KW",
        "BUILD_LIST",
        "BUILD_TUPLE",
        "BUILD_SET",
        "BUILD_MAP",
        "BUILD_CONST_KEY_MAP",
        "BUILD_SLICE",
        "BINARY_SUBSCR",
        "COPY",
        "SWAP",
        "TO_BOOL",
        "LIST_EXTEND",
        "SET_UPDATE",
        "DICT_UPDATE",
        "DICT_MERGE",
        "PYARCH_COMPREHENSION",
        "PYARCH_BOOLOP",
        "RETURN_GENERATOR",
        "YIELD_VALUE",
        "CALL_INTRINSIC_1",
        "CONVERT_VALUE",
        "FORMAT_SIMPLE",
        "FORMAT_WITH_SPEC",
        "BUILD_STRING",
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
        if len(stack) == 0:
            # Implicit cleanup pop (e.g. discarding a for-loop's
            # iterator on `break`/early return) that this per-block
            # model does not track as a real stack value. Safe to
            # treat as a no-op rather than fail.
            return None

        expression = _pop_expression(
            stack
        )

        if isinstance(
            expression, (_ImportRef, _ImportFromRef)
        ):
            # Discards the module object left on the stack after the
            # last `IMPORT_FROM` of a `from ... import ...`
            # statement; it never appears in the source.
            return None

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

        unpack_result = _handle_unpack_slot(value, name)

        if unpack_result is not _NOT_AN_UNPACK:
            return unpack_result

        definition = _resolve_definition(
            value,
            name,
            code_map,
        )

        if definition is not None:
            return definition

        return Assign(
            target=Name(
                name=name
            ),
            value=value,
            offset=instruction.offset,
        )

    if op == "STORE_FAST_STORE_FAST":
        if not (
            isinstance(instruction.argval, tuple)
            and len(instruction.argval) == 2
        ):
            raise StatementError(
                "STORE_FAST_STORE_FAST has invalid names: "
                f"{instruction.argval!r}"
            )

        first_name, second_name = instruction.argval

        # CPython pops in the same order two separate STORE_FASTs
        # would: TOS goes to the first name, the next value to the
        # second.
        first_value = _pop_expression(stack)
        second_value = _pop_expression(stack)

        results: list[IRStatement] = []

        for name, value in (
            (first_name, first_value),
            (second_name, second_value),
        ):
            unpack_result = _handle_unpack_slot(value, name)

            if unpack_result is not _NOT_AN_UNPACK:
                if unpack_result is not None:
                    results.append(unpack_result)
                continue

            definition = _resolve_definition(
                value, name, code_map
            )

            results.append(
                definition
                if definition is not None
                else Assign(
                    target=Name(name=name),
                    value=value,
                    offset=instruction.offset,
                )
            )

        return results

    if op == "STORE_FAST_LOAD_FAST":
        if not (
            isinstance(instruction.argval, tuple)
            and len(instruction.argval) == 2
        ):
            raise StatementError(
                "STORE_FAST_LOAD_FAST has invalid names: "
                f"{instruction.argval!r}"
            )

        store_name, load_name = instruction.argval
        value = _pop_expression(stack)

        definition = _resolve_definition(
            value, store_name, code_map
        )

        statement = (
            definition
            if definition is not None
            else Assign(
                target=Name(name=store_name),
                value=value,
                offset=instruction.offset,
            )
        )

        stack.push(Name(name=load_name))

        return statement

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

        # CPython: "Implements TOS.name = TOS1" -- TOS (popped
        # first) is the object, TOS1 (popped second) is the value.
        object_expression = _pop_expression(
            stack
        )

        value = _pop_expression(
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

        # CPython: "Implements TOS1[TOS] = TOS2" -- pop order is
        # index, then object, then value.
        index = _pop_expression(
            stack
        )

        object_expression = _pop_expression(
            stack
        )

        value = _pop_expression(
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


_NOT_AN_UNPACK = object()


def _handle_unpack_slot(
    value: IRExpression,
    name: str,
):
    """
    If `value` is an `_UnpackSlot` from UNPACK_SEQUENCE/UNPACK_EX,
    record this STORE's target and return either `None` (more slots
    still pending) or the completed `Assign` with a tuple target
    (this was the last slot). Returns the `_NOT_AN_UNPACK` sentinel
    when `value` isn't a slot at all, so the caller falls through to
    ordinary assignment handling.
    """

    if not isinstance(value, _UnpackSlot):
        return _NOT_AN_UNPACK

    target = (
        Starred(value=Name(name=name))
        if value.is_starred
        else Name(name=name)
    )

    value.targets[value.position] = target

    if value.position != len(value.targets) - 1:
        return None

    if any(t is None for t in value.targets):
        raise StatementError(
            "Tuple unpacking left an unfilled target."
        )

    return Assign(
        target=TupleExpr(elements=value.targets),
        value=value.seq_expr,
    )


def _dict_expr_to_mapping(
    value: IRExpression,
) -> dict[str, IRExpression] | None:
    """Convert a dict-shaped expression into a name -> expr mapping."""

    from .ir import DictExpr

    if isinstance(value, DictExpr):
        result = {}

        for key_expr, value_expr in value.entries:
            if isinstance(key_expr, Constant) and isinstance(
                key_expr.value, str
            ):
                result[key_expr.value] = value_expr

        return result or None

    if isinstance(value, Constant) and isinstance(
        value.value, dict
    ):
        return {
            key: Constant(value=item)
            for key, item in value.value.items()
            if isinstance(key, str)
        }

    return None


def _flat_annotation_tuple_to_mapping(
    value: IRExpression,
) -> dict[str, IRExpression] | None:
    """
    Convert the flat ``(name1, type1, name2, type2, ...)`` tuple
    CPython builds for a function's ``__annotations__`` into a
    name -> type-expression mapping.
    """

    from .ir import TupleExpr

    elements = None

    if isinstance(value, TupleExpr):
        elements = value.elements
    elif isinstance(value, Constant) and isinstance(
        value.value, tuple
    ):
        elements = [Constant(value=item) for item in value.value]

    if elements is None or len(elements) % 2 != 0:
        return None

    result = {}

    for index in range(0, len(elements), 2):
        name_expr = elements[index]
        type_expr = elements[index + 1]

        if isinstance(name_expr, Constant) and isinstance(
            name_expr.value, str
        ):
            result[name_expr.value] = type_expr

    return result or None


def _unwrap_decorated_function(
    value: IRExpression,
    code_map: dict[int, "Function | Class"] | None,
) -> "Function | Class | None":
    """
    Recognize ``x = dec(def_or_class)`` (and chains of these, for
    stacked decorators) and return the underlying Function/Class with
    `decorators` populated in source order -- or None if `value`
    doesn't have this shape at all (an ordinary assignment).
    """

    if isinstance(value, _FunctionRef):
        function = (
            code_map.get(id(value.code)) if code_map else None
        )

        if not isinstance(function, Function):
            function = Function(
                name=value.code.co_name, parameters=[]
            )

        return replace(function, decorators=[])

    if (
        isinstance(value, Call)
        and isinstance(value.function, Name)
        and value.function.name == "__build_class__"
        and value.args
        and isinstance(value.args[0], _FunctionRef)
    ):
        code_obj = value.args[0].code
        bases = list(value.args[2:])

        klass = code_map.get(id(code_obj)) if code_map else None

        if not isinstance(klass, Class):
            klass = Class(name=code_obj.co_name)

        return replace(klass, decorators=[], bases=bases)

    if isinstance(value, Call) and value.args:
        inner = _unwrap_decorated_function(
            value.args[0], code_map
        )

        if inner is not None and len(value.args) == 1:
            return replace(
                inner,
                decorators=[value.function, *inner.decorators],
            )

    return None


def _resolve_definition(
    value: IRExpression,
    name: str,
    code_map: dict[int, "Function | Class"] | None,
) -> "Function | Class | Import | ImportFrom | None":
    """
    Detect whether a STORE actually defines a function, a class, or
    completes an import, rather than an ordinary assignment.
    """

    if isinstance(value, _ImportRef):
        top_level = value.module.split(".")[0]

        asname = None if name == top_level else name

        bound_module = (
            value.module if asname is None else value.module
        )

        return Import(
            names=[(bound_module, asname)]
        )

    if isinstance(value, _ImportFromRef):
        asname = None if name == value.name else name

        return ImportFrom(
            module=value.module,
            names=[(value.name, asname)],
            level=value.level,
        )

    if isinstance(value, _FunctionRef):
        function = (
            code_map.get(id(value.code))
            if code_map
            else None
        )

        if not isinstance(function, Function):
            # No pre-built body available (e.g. lambdas assigned to
            # a name): fall back to an empty-body placeholder rather
            # than inventing behaviour.
            function = Function(
                name=value.code.co_name,
                parameters=[],
            )

        defaults_by_name = {}

        if value.defaults:
            plain_names = [
                p
                for p in function.parameters
                if p not in ("*",)
                and not p.startswith("*")
            ]

            trailing = (
                plain_names[-len(value.defaults):]
                if len(value.defaults) <= len(plain_names)
                else plain_names
            )

            defaults_by_name = dict(
                zip(trailing, value.defaults)
            )

        if value.kwdefaults:
            defaults_by_name = {
                **defaults_by_name,
                **value.kwdefaults,
            }

        merged_annotations = (
            value.annotations or function.annotations
        )

        return replace(
            function,
            name=name,
            defaults=defaults_by_name or function.defaults,
            annotations=merged_annotations,
            returns=(
                merged_annotations.get("return")
                if merged_annotations
                else function.returns
            ),
        )

    if isinstance(value, Call):
        decorated = _unwrap_decorated_function(value, code_map)

        if decorated is not None:
            return replace(decorated, name=name)

    if (
        isinstance(value, Call)
        and isinstance(value.function, Name)
        and value.function.name == "__build_class__"
        and value.args
        and isinstance(value.args[0], _FunctionRef)
    ):
        code_obj = value.args[0].code
        bases = list(value.args[2:])

        klass = (
            code_map.get(id(code_obj))
            if code_map
            else None
        )

        if not isinstance(klass, Class):
            klass = Class(name=code_obj.co_name)

        return replace(
            klass,
            name=name,
            bases=bases,
        )

    return None


def reconstruct_statements(
    instructions: list[TInstruction],
    code_map: dict[int, "Function | Class"] | None = None,
) -> StatementResult:
    """
    Reconstruct statements from a linear instruction sequence.

    This function is deliberately conservative. Instructions that
    require control-flow analysis are left for later passes.

    `code_map` maps `id(code_object)` to an already fully-decompiled
    Function/Class so that nested `def`/`class` statements can be
    reconstructed inline instead of failing on MAKE_FUNCTION /
    LOAD_BUILD_CLASS.
    """

    stack = VirtualStack()

    statements: list[IRStatement] = []

    for instruction in instructions:
        statement = process_instruction(
            stack,
            instruction,
            code_map,
        )

        if statement is None:
            continue

        if isinstance(statement, list):
            statements.extend(statement)
        else:
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