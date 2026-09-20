from __future__ import annotations

"""
Virtual Python stack reconstruction.

This module simulates the CPython evaluation stack at a semantic level.

Pipeline:

    TInstruction
         ↓
    VirtualStack
         ↓
    IR expressions
         ↓
    statements / control flow
"""


from dataclasses import dataclass
from typing import Any

from .ir import (
    Attribute,
    BinaryOp,
    Call,
    Compare,
    Constant,
    DictExpr,
    IRExpression,
    ListExpr,
    Name,
    SetExpr,
    SliceExpr,
    Subscript,
    TupleExpr,
    UnaryOp,
)
from .translate import TInstruction


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StackError(Exception):
    """Raised when bytecode stack reconstruction becomes inconsistent."""


# ---------------------------------------------------------------------------
# Stack values
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StackValue:
    """
    A value stored on the virtual Python evaluation stack.

    `expression` contains the reconstructed semantic expression.

    `kind` is mostly diagnostic and lets later passes distinguish
    special stack values such as NULL used by CALL.
    """

    expression: IRExpression | None = None

    kind: str = "value"

    data: Any = None


# ---------------------------------------------------------------------------
# Virtual stack
# ---------------------------------------------------------------------------


class VirtualStack:
    """A semantic simulation of the CPython evaluation stack."""

    def __init__(self) -> None:
        self._values: list[StackValue] = []

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._values)

    def __bool__(self) -> bool:
        return bool(self._values)

    def clear(self) -> None:
        """Clear the stack."""

        self._values.clear()

    def snapshot(self) -> list[StackValue]:
        """Return a shallow copy of the current stack."""

        return list(self._values)

    def restore(
        self,
        values: list[StackValue],
    ) -> None:
        """Restore a previously saved stack."""

        self._values = list(values)

    def push(
        self,
        expression: IRExpression | None = None,
        *,
        kind: str = "value",
        data: Any = None,
    ) -> None:
        """Push a value onto the stack."""

        self._values.append(
            StackValue(
                expression=expression,
                kind=kind,
                data=data,
            )
        )

    def push_value(
        self,
        value: Any,
    ) -> None:
        """Push a Python constant."""

        self.push(
            Constant(value=value)
        )

    def push_name(
        self,
        name: str,
    ) -> None:
        """Push a name reference."""

        self.push(
            Name(name=name)
        )

    def push_null(self) -> None:
        """
        Push CPython's internal NULL call marker.

        This is not emitted as Python source.
        """

        self.push(
            expression=None,
            kind="null",
        )

    def pop(
        self,
    ) -> StackValue:
        """Pop the top value."""

        if not self._values:
            raise StackError(
                "Evaluation stack underflow."
            )

        return self._values.pop()

    def peek(
        self,
        depth: int = 0,
    ) -> StackValue:
        """
        Read a value without removing it.

        depth=0 means the top of the stack.
        """

        index = len(self._values) - 1 - depth

        if index < 0:
            raise StackError(
                "Evaluation stack underflow."
            )

        return self._values[index]

    def pop_expression(
        self,
    ) -> IRExpression:
        """Pop a value and require it to contain an expression."""

        value = self.pop()

        if value.expression is None:
            raise StackError(
                "Expected an expression on the evaluation stack."
            )

        return value.expression

    # ------------------------------------------------------------------
    # Stack manipulation
    # ------------------------------------------------------------------

    def copy(
        self,
        depth: int,
    ) -> None:
        """
        COPY depth.

        Python's COPY uses 1-based depth:
            COPY 1 -> duplicate TOS
            COPY 2 -> duplicate second item
        """

        if depth <= 0:
            raise StackError(
                f"Invalid COPY depth: {depth}"
            )

        value = self.peek(depth - 1)

        self._values.append(
            StackValue(
                expression=value.expression,
                kind=value.kind,
                data=value.data,
            )
        )

    def swap(
        self,
        depth: int,
    ) -> None:
        """
        SWAP depth.

        Python's SWAP uses 1-based depth.
        """

        if depth <= 0:
            raise StackError(
                f"Invalid SWAP depth: {depth}"
            )

        index = len(self._values) - depth

        if index < 0:
            raise StackError(
                "SWAP depth exceeds stack size."
            )

        top_index = len(self._values) - 1

        (
            self._values[index],
            self._values[top_index],
        ) = (
            self._values[top_index],
            self._values[index],
        )

    # ------------------------------------------------------------------
    # Unary operations
    # ------------------------------------------------------------------

    def unary(
        self,
        operator: str,
    ) -> None:
        """Apply a unary operation."""

        operand = self.pop_expression()

        self.push(
            UnaryOp(
                op=operator,
                operand=operand,
            )
        )

    # ------------------------------------------------------------------
    # Binary operations
    # ------------------------------------------------------------------

    def binary(
        self,
        operator: str,
    ) -> None:
        """Apply a binary operation."""

        right = self.pop_expression()
        left = self.pop_expression()

        self.push(
            BinaryOp(
                op=operator,
                left=left,
                right=right,
            )
        )

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def compare(
        self,
        operator: str,
    ) -> None:
        """Apply a comparison."""

        right = self.pop_expression()
        left = self.pop_expression()

        self.push(
            Compare(
                op=operator,
                left=left,
                right=right,
            )
        )

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def load_attribute(
        self,
        name: str,
    ) -> None:
        """Apply attribute access."""

        value = self.pop_expression()

        self.push(
            Attribute(
                value=value,
                name=name,
            )
        )

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscript(
        self,
    ) -> None:
        """Apply subscription: value[index]."""

        index = self.pop_expression()
        value = self.pop_expression()

        self.push(
            Subscript(
                value=value,
                index=index,
            )
        )

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------

    def call_kw(
        self,
        argument_count: int,
    ) -> None:
        """
        Reconstruct a call with keyword arguments (CALL_KW).

        Stack (bottom to top): ... [NULL] function arg1 ... argN
        kwnames_tuple -- the last `len(kwnames)` of the N arguments
        are the keyword arguments, in the same order as kwnames.
        """

        if argument_count < 0:
            raise StackError(
                f"Invalid argument count: {argument_count}"
            )

        kwnames_expr = self.pop_expression()

        if isinstance(kwnames_expr, Constant) and isinstance(
            kwnames_expr.value, tuple
        ):
            kwnames = list(kwnames_expr.value)
        else:
            raise StackError(
                "CALL_KW: expected a constant tuple of keyword "
                "names."
            )

        if len(self) < argument_count + 1:
            raise StackError(
                "Not enough values on stack for CALL_KW."
            )

        arguments: list[IRExpression] = []

        for _ in range(argument_count):
            arguments.append(self.pop_expression())

        arguments.reverse()

        positional_count = argument_count - len(kwnames)

        if positional_count < 0:
            raise StackError(
                "CALL_KW: more keyword names than arguments."
            )

        positional = arguments[:positional_count]
        keyword_values = arguments[positional_count:]

        keywords = list(zip(kwnames, keyword_values))

        function = self.pop()

        if function.kind == "null":
            function = self.pop()
        elif self._values and self._values[-1].kind == "null":
            self.pop()

        if function.expression is None:
            raise StackError(
                "CALL_KW target does not contain an expression."
            )

        self.push(
            Call(
                function=function.expression,
                args=positional,
                keywords=keywords,
            )
        )

    def call(
        self,
        argument_count: int,
    ) -> None:
        """
        Reconstruct a positional function call.

        The stack is expected to contain:

            ... function arg1 arg2 ... argN

        or, for modern CPython:

            ... NULL function arg1 arg2 ... argN

        The NULL marker is removed if present.
        """

        if argument_count < 0:
            raise StackError(
                f"Invalid argument count: {argument_count}"
            )

        if len(self) < argument_count + 1:
            raise StackError(
                "Not enough values on stack for CALL."
            )

        arguments: list[IRExpression] = []

        for _ in range(argument_count):
            arguments.append(
                self.pop_expression()
            )

        arguments.reverse()

        if (
            argument_count == 0
            and self._values
            and self._values[-1].kind != "null"
            and _looks_like_decoratable(
                self._values[-1].expression
            )
            and len(self._values) >= 2
            and self._values[-2].kind != "null"
        ):
            # CPython compiles decorator application
            # (`@dec\ndef f(): ...` / `@dec\nclass C: ...`) as
            # `[dec, value]` with CALL 0 and no NULL marker at all --
            # unlike an ordinary zero-arg call, which always has a
            # NULL immediately below the callable. Recognize the
            # shape here rather than misreading `value` as a
            # zero-arg call and losing `dec` as a dangling value.
            value_slot = self.pop()
            decorator_slot = self.pop()

            if (
                value_slot.expression is not None
                and decorator_slot.expression is not None
            ):
                self.push(
                    Call(
                        function=decorator_slot.expression,
                        args=[value_slot.expression],
                        keywords=[],
                    )
                )
                return

            # Fall through with what we popped restored, in the
            # unlikely case either slot wasn't a real expression.
            self.push(
                decorator_slot.expression,
                kind=decorator_slot.kind,
                data=decorator_slot.data,
            )
            self.push(
                value_slot.expression,
                kind=value_slot.kind,
                data=value_slot.data,
            )

        function = self.pop()

        if function.kind == "null":
            # Older/alternative convention: NULL was popped where we
            # expected the function; the real function is one slot
            # further down.
            function = self.pop()
        elif self._values and self._values[-1].kind == "null":
            # Modern CPython convention: ... NULL function args... --
            # the NULL marker sits just below the function and must
            # be discarded too, or it corrupts later stack reads.
            self.pop()

        if function.expression is None:
            raise StackError(
                "CALL target does not contain an expression."
            )

        self.push(
            Call(
                function=function.expression,
                args=arguments,
            )
        )

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    def build_list(
        self,
        count: int,
    ) -> None:
        """Build a list from the top `count` stack values."""

        elements = self._pop_expressions(
            count
        )

        self.push(
            ListExpr(
                elements=elements
            )
        )

    def build_tuple(
        self,
        count: int,
    ) -> None:
        """Build a tuple from the top `count` stack values."""

        elements = self._pop_expressions(
            count
        )

        self.push(
            TupleExpr(
                elements=elements
            )
        )

    def build_set(
        self,
        count: int,
    ) -> None:
        """Build a set from the top `count` stack values."""

        elements = self._pop_expressions(
            count
        )

        self.push(
            SetExpr(
                elements=elements
            )
        )

    def _literal_elements(
        self,
        expression: IRExpression,
    ) -> list[IRExpression] | None:
        """
        Return the individual elements of an expression when they are
        statically known (a literal container or constant tuple/list/
        set), or None when the expression is opaque (e.g. an
        arbitrary variable being unpacked with ``*``).
        """

        if isinstance(
            expression,
            (ListExpr, TupleExpr, SetExpr),
        ):
            return list(expression.elements)

        if isinstance(
            expression,
            Constant,
        ) and isinstance(
            expression.value,
            (tuple, list, set, frozenset),
        ):
            return [
                Constant(value=item)
                for item in expression.value
            ]

        return None

    def extend_list(
        self,
    ) -> None:
        """
        LIST_EXTEND: extend a list in place with an iterable.

        Used by CPython to build list literals such as ``[1, 2, 3]``
        (``BUILD_LIST 0`` + a constant tuple + ``LIST_EXTEND``) and by
        starred unpacking such as ``[a, *b, c]``.
        """

        source = self.pop_expression()
        target = self.pop_expression()

        if not isinstance(target, ListExpr):
            raise StackError(
                "LIST_EXTEND expected a list on the stack."
            )

        elements = self._literal_elements(source)

        if elements is None:
            raise StackError(
                "LIST_EXTEND: cannot statically unpack a "
                "non-literal iterable (unsupported starred "
                "expression)."
            )

        self.push(
            ListExpr(
                elements=target.elements + elements
            )
        )

    def update_set(
        self,
    ) -> None:
        """SET_UPDATE: extend a set in place with an iterable."""

        source = self.pop_expression()
        target = self.pop_expression()

        if not isinstance(target, SetExpr):
            raise StackError(
                "SET_UPDATE expected a set on the stack."
            )

        elements = self._literal_elements(source)

        if elements is None:
            raise StackError(
                "SET_UPDATE: cannot statically unpack a "
                "non-literal iterable (unsupported starred "
                "expression)."
            )

        self.push(
            SetExpr(
                elements=target.elements + elements
            )
        )

    def update_dict(
        self,
    ) -> None:
        """DICT_UPDATE / DICT_MERGE: merge a mapping into a dict."""

        source = self.pop_expression()
        target = self.pop_expression()

        if not isinstance(target, DictExpr):
            raise StackError(
                "DICT_UPDATE expected a dict on the stack."
            )

        if isinstance(source, DictExpr):
            entries = list(source.entries)
        elif isinstance(
            source, Constant
        ) and isinstance(source.value, dict):
            entries = [
                (Constant(value=key), Constant(value=value))
                for key, value in source.value.items()
            ]
        else:
            raise StackError(
                "DICT_UPDATE: cannot statically unpack a "
                "non-literal mapping (e.g. ``**other``)."
            )

        self.push(
            DictExpr(
                entries=target.entries + entries
            )
        )

    def build_map(
        self,
        count: int,
    ) -> None:
        """
        Build a dictionary.

        BUILD_MAP pushes keys and values in alternating order.
        """

        if count < 0:
            raise StackError(
                f"Invalid map size: {count}"
            )

        entries: list[
            tuple[
                IRExpression | None,
                IRExpression,
            ]
        ] = []

        for _ in range(count):
            value = self.pop_expression()
            key = self.pop_expression()

            entries.append(
                (key, value)
            )

        entries.reverse()

        self.push(
            DictExpr(
                entries=entries
            )
        )

    def build_const_key_map(
        self,
        count: int,
    ) -> None:
        """
        Build a dictionary from constant keys.

        Stack layout:

            ... value1 value2 ... valueN keys_tuple
        """

        keys = self.pop_expression()

        if not isinstance(
            keys,
            TupleExpr,
        ):
            raise StackError(
                "BUILD_CONST_KEY_MAP expected a tuple of keys."
            )

        values = self._pop_expressions(
            count
        )

        if len(keys.elements) != len(values):
            raise StackError(
                "BUILD_CONST_KEY_MAP key/value count mismatch."
            )

        self.push(
            DictExpr(
                entries=[
                    (
                        key,
                        value,
                    )
                    for key, value in zip(
                        keys.elements,
                        values,
                    )
                ]
            )
        )

    # ------------------------------------------------------------------
    # Slice
    # ------------------------------------------------------------------

    def build_slice(
        self,
        count: int,
    ) -> None:
        """Build a slice expression."""

        if count not in {2, 3}:
            raise StackError(
                f"Unsupported BUILD_SLICE count: {count}"
            )

        if count == 2:
            upper = self.pop_expression()
            lower = self.pop_expression()

            self.push(
                SliceExpr(
                    lower=lower,
                    upper=upper,
                )
            )

            return

        step = self.pop_expression()
        upper = self.pop_expression()
        lower = self.pop_expression()

        self.push(
            SliceExpr(
                lower=lower,
                upper=upper,
                step=step,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pop_expressions(
        self,
        count: int,
    ) -> list[IRExpression]:
        """Pop `count` expressions while preserving source order."""

        if count < 0:
            raise StackError(
                f"Invalid item count: {count}"
            )

        if len(self) < count:
            raise StackError(
                "Not enough values on stack."
            )

        values: list[IRExpression] = []

        for _ in range(count):
            values.append(
                self.pop_expression()
            )

        values.reverse()

        return values


# ---------------------------------------------------------------------------
# Instruction helpers
# ---------------------------------------------------------------------------


_BINARY_OPS = {
    "+",
    "-",
    "*",
    "@",
    "/",
    "//",
    "%",
    "**",
    "<<",
    ">>",
    "&",
    "|",
    "^",
}


_COMPARE_OPS = {
    "<",
    "<=",
    "==",
    "!=",
    ">",
    ">=",
    "in",
    "not in",
    "is",
    "is not",
}


def _looks_like_decoratable(
    expression: IRExpression | None,
) -> bool:
    """
    True for the specific shapes that appear right after
    MAKE_FUNCTION or a class body: something a decorator could be
    wrapping. Deliberately narrow -- this is what distinguishes
    CPython's optimized decorator-call shape (no NULL marker at all)
    from an ordinary zero-argument call like ``obj.method()``, whose
    result would otherwise be misread as a decorator target.
    """

    if expression is None:
        return False

    if hasattr(expression, "code"):
        # Duck-typed check for statements.py's `_FunctionRef` without
        # importing it here (that module imports this one).
        return True

    if (
        isinstance(expression, Call)
        and isinstance(expression.function, Name)
        and expression.function.name == "__build_class__"
    ):
        return True

    return False


def apply_instruction(
    stack: VirtualStack,
    instruction: TInstruction,
) -> None:
    """
    Apply one normalized instruction to the virtual stack.

    Only stack semantics are handled here.

    Statements such as STORE_NAME and RETURN_VALUE are intentionally
    handled by higher-level passes.
    """

    op = instruction.op

    if op in {
        "RESUME",
        "NOP",
        "PRECALL",
        "KW_NAMES",
    }:
        return

    if op == "LOAD_CONST":
        stack.push(
            Constant(
                value=instruction.argval
            )
        )
        return

    if op in {
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "LOAD_FAST",
        "LOAD_FAST_CHECK",
        "LOAD_DEREF",
        "LOAD_CLOSURE",
    }:
        if not isinstance(
            instruction.argval,
            str,
        ):
            raise StackError(
                f"{op} has invalid name: "
                f"{instruction.argval!r}"
            )

        stack.push_name(
            instruction.argval
        )

        return

    if op == "LOAD_FAST_LOAD_FAST":
        if not (
            isinstance(instruction.argval, tuple)
            and len(instruction.argval) == 2
        ):
            raise StackError(
                "LOAD_FAST_LOAD_FAST has invalid names: "
                f"{instruction.argval!r}"
            )

        first_name, second_name = instruction.argval
        stack.push_name(first_name)
        stack.push_name(second_name)
        return

    if op == "PUSH_NULL":
        stack.push_null()
        return

    if op == "LOAD_SUPER_ATTR":
        if not isinstance(
            instruction.argval,
            str,
        ):
            raise StackError(
                "LOAD_SUPER_ATTR has invalid attribute name."
            )

        try:
            stack.pop_expression()  # self (unused: super() is 0-arg)
            stack.pop_expression()  # __class__ cell (unused)
            stack.pop_expression()  # the `super` global itself
        except StackError as error:
            raise StackError(
                f"LOAD_SUPER_ATTR: {error}"
            ) from error

        stack.push(
            Attribute(
                value=Call(
                    function=Name(name="super"),
                    args=[],
                    keywords=[],
                ),
                name=instruction.argval,
            )
        )

        return

    if op == "LOAD_ATTR":
        if not isinstance(
            instruction.argval,
            str,
        ):
            raise StackError(
                "LOAD_ATTR has invalid attribute name."
            )

        stack.load_attribute(
            instruction.argval
        )

        return

    if op == "BINARY_OP":
        operator = instruction.argval

        if not isinstance(
            operator,
            str,
        ):
            operator = instruction.argrepr

        if operator not in _BINARY_OPS:
            raise StackError(
                f"Unsupported binary operator: "
                f"{operator!r}"
            )

        stack.binary(
            operator
        )

        return

    if op in {
        "UNARY_NOT",
        "UNARY_NEGATIVE",
        "UNARY_POSITIVE",
        "UNARY_INVERT",
    }:
        operators = {
            "UNARY_NOT": "not",
            "UNARY_NEGATIVE": "-",
            "UNARY_POSITIVE": "+",
            "UNARY_INVERT": "~",
        }

        stack.unary(
            operators[op]
        )

        return

    if op == "COMPARE_OP":
        operator = instruction.argrepr

        if operator not in _COMPARE_OPS:
            raise StackError(
                f"Unsupported comparison: "
                f"{operator!r}"
            )

        stack.compare(
            operator
        )

        return

    if op == "IS_OP":
        operator = (
            "is not"
            if instruction.arg
            else "is"
        )

        stack.compare(
            operator
        )

        return

    if op == "CONTAINS_OP":
        operator = (
            "not in"
            if instruction.arg
            else "in"
        )

        stack.compare(
            operator
        )

        return

    if op == "CALL":
        if instruction.arg is None:
            raise StackError(
                "CALL has no argument count."
            )

        stack.call(
            instruction.arg
        )

        return

    if op == "CALL_KW":
        if instruction.arg is None:
            raise StackError(
                "CALL_KW has no argument count."
            )

        stack.call_kw(
            instruction.arg
        )

        return

    if op == "BUILD_LIST":
        stack.build_list(
            instruction.arg or 0
        )
        return

    if op == "BUILD_TUPLE":
        stack.build_tuple(
            instruction.arg or 0
        )
        return

    if op == "BUILD_SET":
        stack.build_set(
            instruction.arg or 0
        )
        return

    if op == "BUILD_MAP":
        stack.build_map(
            instruction.arg or 0
        )
        return

    if op == "PYARCH_BOOLOP":
        from .ir import BoolOp

        kind, right_expr = instruction.argval
        left_expr = stack.pop_expression()
        stack.push(
            BoolOp(op=kind, values=[left_expr, right_expr])
        )
        return

    if op == "PYARCH_COMPREHENSION":
        stack.push(instruction.argval)
        return

    if op == "TO_BOOL":
        # Coerces TOS to bool for the upcoming conditional jump.
        # Source reconstruction doesn't need to represent this
        # explicitly -- Python evaluates truthiness the same way
        # regardless -- so it's a no-op on the expression itself.
        return

    if op == "LIST_EXTEND":
        stack.extend_list()
        return

    if op == "SET_UPDATE":
        stack.update_set()
        return

    if op in {
        "DICT_UPDATE",
        "DICT_MERGE",
    }:
        stack.update_dict()
        return

    if op == "BUILD_CONST_KEY_MAP":
        stack.build_const_key_map(
            instruction.arg or 0
        )
        return

    if op == "BUILD_SLICE":
        stack.build_slice(
            instruction.arg or 0
        )
        return

    if op == "BINARY_SUBSCR":
        stack.subscript()
        return

    if op == "POP_TOP":
        stack.pop()
        return

    if op == "RETURN_GENERATOR":
        # Marks the function as a generator; the value it pushes is
        # immediately discarded by the POP_TOP CPython always emits
        # right after it. No source-level representation needed.
        return

    if op == "YIELD_VALUE":
        from .ir import Yield

        value = stack.pop_expression()
        stack.push(Yield(value=value))
        return

    if op == "CALL_INTRINSIC_1":
        from .ir import ListExpr, TupleExpr

        # INTRINSIC_LIST_TO_TUPLE (used when building the final
        # positional-args tuple for a call with `*iterable`
        # unpacking): fold the list literal into a tuple literal
        # rather than modelling the conversion as a call.
        if instruction.argval == "INTRINSIC_LIST_TO_TUPLE" or (
            instruction.arg == 6
        ):
            value = stack.pop_expression()

            if isinstance(value, ListExpr):
                stack.push(TupleExpr(elements=value.elements))
            else:
                stack.push(value)

            return

        raise StackError(
            f"Unsupported CALL_INTRINSIC_1: {instruction.argval!r}"
        )

    if op == "COPY":
        if instruction.arg is None:
            raise StackError(
                "COPY has no depth."
            )

        stack.copy(
            instruction.arg
        )

        return

    if op == "CONVERT_VALUE":
        from .ir import FormattedValue

        conversion_map = {1: 115, 2: 114, 3: 97}
        conversion = conversion_map.get(instruction.arg, -1)

        value = stack.pop_expression()

        stack.push(
            FormattedValue(
                value=value,
                conversion=conversion,
                format_spec=None,
            )
        )
        return

    if op == "FORMAT_SIMPLE":
        from .ir import FormattedValue

        value = stack.pop_expression()

        if isinstance(value, FormattedValue):
            stack.push(value)
        else:
            stack.push(
                FormattedValue(
                    value=value,
                    conversion=-1,
                    format_spec=None,
                )
            )
        return

    if op == "FORMAT_WITH_SPEC":
        from .ir import FormattedValue

        format_spec = stack.pop_expression()
        value = stack.pop_expression()

        if isinstance(value, FormattedValue):
            value.format_spec = format_spec
            stack.push(value)
        else:
            stack.push(
                FormattedValue(
                    value=value,
                    conversion=-1,
                    format_spec=format_spec,
                )
            )
        return

    if op == "BUILD_STRING":
        from .ir import FormattedValue, JoinedStr

        count = instruction.arg or 0
        parts = [stack.pop_expression() for _ in range(count)]
        parts.reverse()

        # A JoinedStr with no FormattedValue parts is just adjacent
        # string literal concatenation (e.g. implicit `"a" "b"` or
        # plain str(...)-building code, not necessarily from an
        # f-string) -- fold it into a single Constant when every part
        # is a plain string constant, which is both simpler and more
        # faithful to what the source likely was.
        if all(
            isinstance(part, Constant)
            and isinstance(part.value, str)
            for part in parts
        ):
            stack.push(
                Constant(
                    value="".join(part.value for part in parts)
                )
            )
        else:
            stack.push(JoinedStr(values=parts))
        return

    if op == "SWAP":
        if instruction.arg is None:
            raise StackError(
                "SWAP has no depth."
            )

        if instruction.arg <= len(stack):
            stack.swap(
                instruction.arg
            )
        # else: this SWAP is protecting a real value across the
        # exception-handling machinery (e.g. a `return` inside an
        # `except` block, swapped past state this model doesn't
        # track as a value). Harmless to skip: the value we do
        # track keeps its position, which is all that matters for
        # source reconstruction.

        return

    # Instructions that manipulate program state rather than expressions
    # are deliberately ignored by this layer.
    if op in {
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_FAST",
        "STORE_DEREF",
        "DELETE_NAME",
        "DELETE_GLOBAL",
        "DELETE_FAST",
        "DELETE_DEREF",
        "STORE_ATTR",
        "DELETE_ATTR",
        "STORE_SUBSCR",
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
        "FOR_ITER",
        "GET_ITER",
        "END_FOR",
        "RETURN_VALUE",
        "RETURN_CONST",
        "RAISE_VARARGS",
        "RERAISE",
    }:
        return

    raise StackError(
        f"Unsupported stack instruction: {op}"
    )


def simulate(
    instructions: list[TInstruction],
) -> VirtualStack:
    """
    Simulate a linear sequence of instructions.

    This helper is primarily useful for simple code objects and tests.

    Full control-flow-aware simulation will be implemented by the CFG
    and decompiler engine later.
    """

    stack = VirtualStack()

    for instruction in instructions:
        apply_instruction(
            stack,
            instruction,
        )

    return stack