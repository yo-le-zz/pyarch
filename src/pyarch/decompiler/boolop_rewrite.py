from __future__ import annotations

"""
Reconstruction of `and`/`or` used as an *expression* value (e.g.
`return a and b`, `x = a or b`), as opposed to only steering control
flow in an `if`/`while` test.

CPython compiles this as: evaluate the left operand, COPY it,
optionally TO_BOOL it, then a conditional jump that -- on the
short-circuiting outcome -- jumps straight past the right operand,
leaving the duplicated left value as the expression's result; on the
other outcome, it pops the duplicate, evaluates the right operand,
and that becomes the result instead.

This is fundamentally different from an `if`-test's conditional jump
(there, the value is fully consumed, not partially kept around for a
later merge), so it's handled as its own instruction-stream rewrite,
the same way inlined comprehensions are: replace the whole
COPY..right-operand span with one synthetic instruction. Unlike
comprehensions, the left operand is *not* part of the rewritten span
-- it was already computed by whatever precedes COPY, and stays on
the stack for the synthetic instruction to consume normally.
"""

from dataclasses import dataclass

from .ir import BoolOp
from .stack import StackError, VirtualStack, apply_instruction
from .translate import TInstruction


class BoolOpRewriteError(Exception):
    """Raised internally when a candidate shape doesn't match."""


def _index(
    instructions: list[TInstruction],
) -> tuple[dict[int, TInstruction], list[int]]:
    by_offset = {i.offset: i for i in instructions}
    return by_offset, sorted(by_offset)


def _next(sorted_offsets: list[int], offset: int) -> int | None:
    import bisect

    index = bisect.bisect_right(sorted_offsets, offset)
    return sorted_offsets[index] if index < len(sorted_offsets) else None


def _mini_eval(instructions: list[TInstruction]):
    stack = VirtualStack()

    for instruction in instructions:
        try:
            apply_instruction(stack, instruction)
        except StackError as error:
            raise BoolOpRewriteError(
                f"could not evaluate right operand: {error}"
            ) from error

    try:
        return stack.pop_expression()
    except StackError as error:
        raise BoolOpRewriteError(
            f"right operand left no value: {error}"
        ) from error


@dataclass(slots=True)
class _Match:
    start_offset: int
    end_offset: int  # exclusive
    op: str
    right_expr: object


def _try_match_at(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    copy_offset: int,
) -> _Match:
    copy_instr = by_offset[copy_offset]

    if copy_instr.arg != 1:
        raise BoolOpRewriteError("COPY depth is not 1")

    offset = _next(sorted_offsets, copy_offset)
    instr = by_offset.get(offset)

    if instr is not None and instr.op == "TO_BOOL":
        offset = _next(sorted_offsets, offset)
        instr = by_offset.get(offset)

    if instr is None or "IF_" not in instr.op:
        raise BoolOpRewriteError(
            "no conditional jump after COPY"
        )

    if "IF_FALSE" in instr.op:
        op = "and"
    elif "IF_TRUE" in instr.op:
        op = "or"
    else:
        raise BoolOpRewriteError(
            f"unsupported conditional jump: {instr.op}"
        )

    target = instr.target

    if not isinstance(target, int):
        raise BoolOpRewriteError("jump has no numeric target")

    offset = _next(sorted_offsets, offset)
    instr = by_offset.get(offset)

    if instr is None or instr.op != "POP_TOP":
        raise BoolOpRewriteError(
            "expected POP_TOP after the short-circuit jump"
        )

    right_start = _next(sorted_offsets, offset)

    right_instrs = [
        by_offset[o]
        for o in sorted_offsets
        if right_start is not None
        and right_start <= o < target
    ]

    right_expr = _mini_eval(right_instrs)

    return _Match(
        start_offset=copy_offset,
        end_offset=target,
        op=op,
        right_expr=right_expr,
    )


def rewrite_boolops(
    instructions: list[TInstruction],
    diagnostics,
) -> list[TInstruction]:
    """
    Scan `instructions` for the and/or-as-expression shape and
    replace each with one synthetic instruction. Code with no such
    expressions (the vast majority, since `if`/`while` tests don't
    go through this path) passes through unchanged.
    """

    by_offset, sorted_offsets = _index(instructions)

    matches: list[_Match] = []

    for offset in sorted_offsets:
        if by_offset[offset].op != "COPY":
            continue

        if any(
            m.start_offset <= offset < m.end_offset for m in matches
        ):
            continue

        try:
            match = _try_match_at(by_offset, sorted_offsets, offset)
        except BoolOpRewriteError:
            # Not every COPY is part of this shape (COPY is used
            # elsewhere too, e.g. dict/attribute augmented
            # assignment) -- silently skip non-matches rather than
            # warning about something that was never broken.
            continue

        matches.append(match)

    if not matches:
        return instructions

    matches_by_start = {m.start_offset: m for m in matches}

    result: list[TInstruction] = []
    skip_until = -1

    for instruction in instructions:
        if instruction.offset < skip_until:
            continue

        match = matches_by_start.get(instruction.offset)

        if match is not None:
            result.append(
                TInstruction(
                    offset=match.start_offset,
                    op="PYARCH_BOOLOP",
                    arg=None,
                    argval=(match.op, match.right_expr),
                    argrepr=match.op,
                    target=None,
                    original="PYARCH_BOOLOP",
                )
            )
            skip_until = match.end_offset
            continue

        result.append(instruction)

    return result
