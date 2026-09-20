from __future__ import annotations

"""
Reconstruction of Python 3.12+ "inlined" comprehensions (PEP 709).

Since 3.12, CPython no longer compiles `[x for x in y]` into a
separate `<listcomp>` code object; it inlines the loop directly into
the surrounding code, using `LOAD_FAST_AND_CLEAR`/`SWAP` to save and
restore a same-named enclosing variable around the comprehension's
own scope.

This module scans the flat instruction stream for that shape and
replaces the whole matched range with a single synthetic
`PYARCH_COMPREHENSION` instruction whose `argval` is the fully-built
`ListComp`/`SetComp`/`DictComp` IR node -- so the rest of the pipeline
(which processes instructions one at a time) can treat a
comprehension exactly like any other expression-producing opcode.

Deliberately conservative: a single `for` clause with at most one
`if` filter. Anything else (nested `for` clauses, `async for`,
generator expressions used lazily) is left alone and reported as
unsupported rather than guessed at.
"""

from dataclasses import dataclass

from .ir import (
    Comprehension,
    DictComp,
    IRExpression,
    ListComp,
    Name,
    SetComp,
)
from .stack import StackError, VirtualStack, apply_instruction
from .translate import TInstruction

_APPEND_OPS = {
    "LIST_APPEND": "list",
    "SET_ADD": "set",
    "MAP_ADD": "dict",
}

_BUILD_OPS = {
    "BUILD_LIST": "list",
    "BUILD_SET": "set",
    "BUILD_MAP": "dict",
}


class ComprehensionRewriteError(Exception):
    """Raised internally when a candidate shape doesn't match."""


def _index(
    instructions: list[TInstruction],
) -> tuple[dict[int, TInstruction], list[int]]:
    by_offset = {i.offset: i for i in instructions}
    return by_offset, sorted(by_offset)


def _next(
    sorted_offsets: list[int],
    offset: int,
) -> int | None:
    import bisect

    index = bisect.bisect_right(sorted_offsets, offset)
    return (
        sorted_offsets[index]
        if index < len(sorted_offsets)
        else None
    )


def _expect(
    by_offset: dict[int, TInstruction],
    offset: int | None,
    *ops: str,
) -> TInstruction:
    instr = by_offset.get(offset) if offset is not None else None

    if instr is None or instr.op not in ops:
        raise ComprehensionRewriteError(
            f"expected one of {ops} at offset {offset}, "
            f"found {instr.op if instr else None}"
        )

    return instr


def _mini_eval(
    instructions: list[TInstruction],
) -> IRExpression:
    stack = VirtualStack()

    for instruction in instructions:
        try:
            apply_instruction(stack, instruction)
        except StackError as error:
            raise ComprehensionRewriteError(
                f"could not evaluate sub-expression: {error}"
            ) from error

    try:
        return stack.pop_expression()
    except StackError as error:
        raise ComprehensionRewriteError(
            f"sub-expression left no value: {error}"
        ) from error


@dataclass(slots=True)
class _Match:
    start_offset: int
    end_offset: int  # exclusive
    expression: IRExpression
    keep_instruction: TInstruction | None = None


def _try_match_at(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    clear_offset: int,
) -> _Match:
    """
    Attempt to match a full inlined-comprehension shape starting at a
    `LOAD_FAST_AND_CLEAR`. Raises `ComprehensionRewriteError` on any
    mismatch (the caller treats that offset as an ordinary
    instruction instead).
    """

    clear_instr = by_offset[clear_offset]
    loop_var = clear_instr.argval

    if not isinstance(loop_var, str):
        raise ComprehensionRewriteError("non-string loop variable")

    offset = _next(sorted_offsets, clear_offset)
    _expect(by_offset, offset, "SWAP")

    offset = _next(sorted_offsets, offset)
    build_instr = _expect(
        by_offset, offset, "BUILD_LIST", "BUILD_SET", "BUILD_MAP"
    )
    kind = _BUILD_OPS[build_instr.op]

    offset = _next(sorted_offsets, offset)
    _expect(by_offset, offset, "SWAP")

    offset = _next(sorted_offsets, offset)
    lookahead = by_offset.get(offset)

    if lookahead is not None and lookahead.op == "GET_ITER":
        # Present in 3.13's shape; 3.12 reuses the iterator set up
        # before LOAD_FAST_AND_CLEAR and goes straight to FOR_ITER.
        offset = _next(sorted_offsets, offset)

    for_iter = _expect(by_offset, offset, "FOR_ITER")
    exit_target = for_iter.target

    header_offset = offset

    offset = _next(sorted_offsets, offset)
    store_instr = _expect(
        by_offset, offset, "STORE_FAST", "STORE_FAST_LOAD_FAST"
    )

    if store_instr.op == "STORE_FAST_LOAD_FAST" and isinstance(
        store_instr.argval, tuple
    ):
        target_name = store_instr.argval[0]
    else:
        target_name = store_instr.argval

    if not isinstance(target_name, str):
        raise ComprehensionRewriteError("non-string target name")

    offset = _next(sorted_offsets, offset)

    # Optional `if` filter: a run of expression instructions ending
    # in POP_JUMP_IF_TRUE (jump forward to the element/append code)
    # with a JUMP_BACKWARD immediately after (the "condition false"
    # path, skipping straight back to the loop header).
    filter_expr = None
    scan = offset

    while True:
        instr = by_offset.get(scan)

        if instr is None:
            raise ComprehensionRewriteError("ran out of instructions")

        if instr.op in _APPEND_OPS or "IF_TRUE" in instr.op:
            break

        scan = _next(sorted_offsets, scan)

    if "IF_TRUE" in by_offset[scan].op:
        filter_instrs = [
            by_offset[o]
            for o in sorted_offsets
            if offset <= o < scan
        ]
        filter_expr = _mini_eval(filter_instrs)

        jump_offset = scan
        after_jump = _next(sorted_offsets, jump_offset)
        _expect(by_offset, after_jump, "JUMP_BACKWARD")

        offset = _next(sorted_offsets, after_jump)
    # else: no filter, `offset` already points at the element/key
    # expression.

    # Element (list/set) or key+value (dict) expression, up to the
    # append opcode.
    append_scan = offset

    while by_offset.get(append_scan) is not None and (
        by_offset[append_scan].op not in _APPEND_OPS
    ):
        append_scan = _next(sorted_offsets, append_scan)

    append_instr = by_offset.get(append_scan)

    if append_instr is None or _APPEND_OPS.get(append_instr.op) != kind:
        raise ComprehensionRewriteError(
            "append opcode missing or doesn't match container kind"
        )

    element_instrs = [
        by_offset[o] for o in sorted_offsets if offset <= o < append_scan
    ]

    if kind == "dict":
        # MAP_ADD consumes key then value from a pair of expressions;
        # CPython emits the key expression, then the value
        # expression. Split by evaluating both against one stack.
        stack = VirtualStack()

        for instr in element_instrs:
            try:
                apply_instruction(stack, instr)
            except StackError as error:
                raise ComprehensionRewriteError(
                    f"dict comprehension key/value: {error}"
                ) from error

        try:
            value_expr = stack.pop_expression()
            key_expr = stack.pop_expression()
        except StackError as error:
            raise ComprehensionRewriteError(
                f"dict comprehension key/value: {error}"
            ) from error

        element_expr: object = (key_expr, value_expr)
    else:
        element_expr = _mini_eval(element_instrs)

    offset = _next(sorted_offsets, append_scan)
    _expect(by_offset, offset, "JUMP_BACKWARD")

    # Loop exit: END_FOR, optional POP_TOP (3.13), optional SWAP
    # (module/class scope), STORE_FAST restoring the saved variable.
    offset = exit_target
    _expect(by_offset, offset, "END_FOR")

    offset = _next(sorted_offsets, offset)
    tail_instr = by_offset.get(offset)

    if tail_instr is not None and tail_instr.op == "POP_TOP":
        offset = _next(sorted_offsets, offset)
        tail_instr = by_offset.get(offset)

    keep_instruction = None

    if tail_instr is not None and tail_instr.op == "SWAP":
        # The list is protected across the restore and remains as
        # this expression's value for whatever follows.
        offset = _next(sorted_offsets, offset)
        restore_instr = _expect(by_offset, offset, "STORE_FAST")

        if restore_instr.argval != loop_var:
            raise ComprehensionRewriteError(
                "restore target does not match the saved variable"
            )

        end_offset = _next(sorted_offsets, offset)
    elif tail_instr is not None and tail_instr.op in (
        "STORE_FAST",
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_DEREF",
    ):
        # The compiler fused "consume the comprehension result" with
        # the restore: the very next store consumes the list (e.g.
        # `values = [...]`), immediately followed by the restore of
        # the cleared variable. Keep the first store (it belongs to
        # the enclosing statement) and drop the second (pure
        # housekeeping, never visible in source).
        keep_instruction = tail_instr
        offset = _next(sorted_offsets, offset)
        restore_instr = _expect(by_offset, offset, "STORE_FAST")

        if restore_instr.argval != loop_var:
            raise ComprehensionRewriteError(
                "restore target does not match the saved variable"
            )

        end_offset = _next(sorted_offsets, offset)
    else:
        raise ComprehensionRewriteError(
            f"unexpected instruction after the loop: "
            f"{tail_instr.op if tail_instr else None}"
        )

    # The iterable expression: everything the FOR_ITER/GET_ITER pair
    # consumed, i.e. between wherever this candidate range logically
    # starts (the caller passes that) and the LOAD_FAST_AND_CLEAR.
    # Reconstructed by the caller, which knows where the range began.

    generator = Comprehension(
        target=Name(name=target_name),
        iter=Name(name="__pyarch_iter__"),
        ifs=[filter_expr] if filter_expr is not None else [],
    )

    if kind == "list":
        expression: IRExpression = ListComp(
            element=element_expr, generators=[generator]
        )
    elif kind == "set":
        expression = SetComp(
            element=element_expr, generators=[generator]
        )
    else:
        key_expr, value_expr = element_expr
        expression = DictComp(
            key=key_expr,
            value=value_expr,
            generators=[generator],
        )

    return _Match(
        start_offset=clear_offset,
        end_offset=end_offset if end_offset is not None else 1 << 30,
        expression=expression,
        keep_instruction=keep_instruction,
    )


def rewrite_comprehensions(
    instructions: list[TInstruction],
    diagnostics,
) -> list[TInstruction]:
    """
    Scan `instructions` for inlined-comprehension shapes and replace
    each with one synthetic instruction. Safe to call unconditionally
    -- code with no comprehensions passes through unchanged, and any
    shape this doesn't recognize is left as-is (surfacing later as an
    "unsupported instruction" diagnostic instead of being silently
    wrong).
    """

    by_offset, sorted_offsets = _index(instructions)

    matches: list[_Match] = []

    for offset in sorted_offsets:
        if by_offset[offset].op != "LOAD_FAST_AND_CLEAR":
            continue

        if any(
            m.start_offset <= offset < m.end_offset for m in matches
        ):
            continue

        try:
            match = _try_match_at(by_offset, sorted_offsets, offset)
        except ComprehensionRewriteError as error:
            diagnostics.warn(
                f"Comprehension at offset {offset} not "
                f"reconstructed (unsupported shape: {error}); it "
                f"will surface as an unsupported instruction."
            )
            continue

        matches.append(match)

    if not matches:
        return instructions

    # The iterable expression precedes each match's own start: it is
    # whatever instructions sit between the previous boundary and
    # `LOAD_FAST_AND_CLEAR`, ending in GET_ITER. Extract it now that
    # we know exactly where each comprehension starts.
    result: list[TInstruction] = []
    covered_until = -1

    for match in sorted(matches, key=lambda m: m.start_offset):
        if match.start_offset <= covered_until:
            continue

        # Find the GET_ITER immediately preceding this match by
        # walking backward from `covered_until`'s successor.
        pre_instrs = [
            by_offset[o]
            for o in sorted_offsets
            if covered_until < o < match.start_offset
        ]

        if not pre_instrs or pre_instrs[-1].op != "GET_ITER":
            diagnostics.warn(
                f"Comprehension at offset {match.start_offset}: "
                f"could not isolate its iterable expression; left "
                f"unreconstructed."
            )
            continue

        try:
            iterable_expr = _mini_eval(pre_instrs[:-1])
        except ComprehensionRewriteError as error:
            diagnostics.warn(
                f"Comprehension at offset {match.start_offset}: "
                f"could not evaluate its iterable: {error}"
            )
            continue

        match.expression.generators[0].iter = iterable_expr

        result.append(
            TInstruction(
                offset=match.start_offset,
                op="PYARCH_COMPREHENSION",
                arg=None,
                argval=match.expression,
                argrepr=repr(match.expression),
                target=None,
                original="PYARCH_COMPREHENSION",
            )
        )

        covered_until = match.end_offset - 1

    # Merge: replace each matched range with its synthetic
    # instruction, keep everything else in original order.
    synthetic_by_start = {
        instr.offset: instr for instr in result
    }

    matches_by_start = {
        match.start_offset: match
        for match in matches
        if match.start_offset in synthetic_by_start
    }

    final: list[TInstruction] = []
    skip_until = -1

    for instruction in instructions:
        if instruction.offset < skip_until:
            continue

        match = matches_by_start.get(instruction.offset)

        if match is not None:
            final.append(synthetic_by_start[match.start_offset])

            if match.keep_instruction is not None:
                final.append(match.keep_instruction)

            skip_until = match.end_offset
            continue

        final.append(instruction)

    return final
