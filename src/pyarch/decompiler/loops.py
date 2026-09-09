from __future__ import annotations

"""
Loop reconstruction for PyArch.

This module detects common Python loop patterns from the normalized
control-flow graph.

It intentionally does not try to reconstruct every possible loop yet.
The goal is to provide reliable structural information to the later
decompiler engine.
"""

from dataclasses import dataclass
from enum import Enum

from .ir import (
    BasicBlock,
    ControlFlowGraph,
    IRInstruction,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LoopError(Exception):
    """Raised when loop reconstruction fails."""


# ---------------------------------------------------------------------------
# Loop types
# ---------------------------------------------------------------------------


class LoopType(str, Enum):
    """Supported loop structures."""

    WHILE = "while"
    FOR = "for"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Loop representation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Loop:
    """
    A reconstructed loop.

    `header` is the block where the loop condition/iteration starts.

    `body` contains blocks belonging to the loop body.

    `exit` is the block reached after the loop, when it can be
    determined reliably.
    """

    loop_type: LoopType

    header: BasicBlock

    body: list[BasicBlock]

    exit: BasicBlock | None = None

    condition: IRInstruction | None = None

    iterator: IRInstruction | None = None

    target: IRInstruction | None = None


# ---------------------------------------------------------------------------
# Basic block helpers
# ---------------------------------------------------------------------------


def _block_offsets(
    block: BasicBlock,
) -> tuple[int, int] | None:
    """Return the first and last instruction offsets."""

    if not block.instructions:
        return None

    return (
        block.instructions[0].offset,
        block.instructions[-1].offset,
    )


def _contains_offset(
    block: BasicBlock,
    offset: int,
) -> bool:
    """Return whether a block contains an instruction offset."""

    bounds = _block_offsets(block)

    if bounds is None:
        return False

    first, last = bounds

    return first <= offset <= last


def _find_block(
    cfg: ControlFlowGraph,
    offset: int,
) -> BasicBlock | None:
    """Find the block containing an instruction offset."""

    for block in cfg.blocks:
        if _contains_offset(
            block,
            offset,
        ):
            return block

    return None


# ---------------------------------------------------------------------------
# Back edges
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BackEdge:
    """A CFG edge that points backwards in bytecode order."""

    source: BasicBlock
    target: BasicBlock


def find_back_edges(
    cfg: ControlFlowGraph,
) -> list[BackEdge]:
    """
    Find backwards CFG edges.

    A backwards edge is a strong indication of a loop because Python
    bytecode normally jumps to a lower instruction offset when
    returning to a loop header.
    """

    edges: list[BackEdge] = []

    for block in cfg.blocks:
        source_bounds = _block_offsets(block)

        if source_bounds is None:
            continue

        source_offset = source_bounds[0]

        for successor in block.successors:
            target_bounds = _block_offsets(
                successor
            )

            if target_bounds is None:
                continue

            target_offset = target_bounds[0]

            if target_offset <= source_offset:
                edges.append(
                    BackEdge(
                        source=block,
                        target=successor,
                    )
                )

    return edges


# ---------------------------------------------------------------------------
# Loop body discovery
# ---------------------------------------------------------------------------


def _block_order(
    cfg: ControlFlowGraph,
) -> dict[int, int]:
    """Return block identity -> bytecode order."""

    return {
        id(block): index
        for index, block in enumerate(
            cfg.blocks
        )
    }


def _predecessors_inside_region(
    block: BasicBlock,
    region: set[int],
) -> list[BasicBlock]:
    """Return predecessors that are part of a loop region."""

    return [
        predecessor
        for predecessor in block.predecessors
        if id(predecessor) in region
    ]


def _collect_loop_body(
    header: BasicBlock,
    source: BasicBlock,
    cfg: ControlFlowGraph,
) -> list[BasicBlock]:
    """
    Collect blocks belonging to a natural loop.

    The algorithm walks backwards through predecessors from the
    back-edge source until it reaches the header.
    """

    region: set[int] = {
        id(header),
        id(source),
    }

    worklist: list[BasicBlock] = [
        source
    ]

    while worklist:
        block = worklist.pop()

        for predecessor in block.predecessors:
            if id(predecessor) in region:
                continue

            region.add(
                id(predecessor)
            )

            if predecessor is not header:
                worklist.append(
                    predecessor
                )

    order = _block_order(cfg)

    return sorted(
        (
            block
            for block in cfg.blocks
            if id(block) in region
        ),
        key=lambda block: order[id(block)],
    )


# ---------------------------------------------------------------------------
# Exit detection
# ---------------------------------------------------------------------------


def _find_loop_exit(
    body: list[BasicBlock],
) -> BasicBlock | None:
    """Find the first successor leaving the loop."""

    body_ids = {
        id(block)
        for block in body
    }

    candidates: list[BasicBlock] = []

    for block in body:
        for successor in block.successors:
            if id(successor) not in body_ids:
                candidates.append(
                    successor
                )

    if not candidates:
        return None

    return candidates[0]


# ---------------------------------------------------------------------------
# FOR detection
# ---------------------------------------------------------------------------


_FOR_ITER_OPS = {
    "FOR_ITER",
}


def _find_instruction(
    block: BasicBlock,
    operations: set[str],
) -> IRInstruction | None:
    """Find the first instruction using one of the supplied operations."""

    for instruction in block.instructions:
        if instruction.op in operations:
            return instruction

    return None


def detect_for_loop(
    header: BasicBlock,
    body: list[BasicBlock],
) -> bool:
    """
    Detect the common bytecode pattern used by Python `for` loops.

    Python versions differ in the exact instructions surrounding
    FOR_ITER, but FOR_ITER itself remains a useful semantic marker
    after translation.
    """

    return (
        _find_instruction(
            header,
            _FOR_ITER_OPS,
        )
        is not None
    )


# ---------------------------------------------------------------------------
# WHILE detection
# ---------------------------------------------------------------------------


_CONDITIONAL_JUMPS = {
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_TRUE_OR_POP",
}


def detect_while_loop(
    header: BasicBlock,
    body: list[BasicBlock],
) -> bool:
    """
    Detect a conditional loop header.

    This intentionally uses structural information rather than
    assuming one exact CPython bytecode sequence.
    """

    instruction = _find_instruction(
        header,
        _CONDITIONAL_JUMPS,
    )

    if instruction is None:
        return False

    return True


# ---------------------------------------------------------------------------
# Loop reconstruction
# ---------------------------------------------------------------------------


def reconstruct_loops(
    cfg: ControlFlowGraph,
) -> list[Loop]:
    """
    Reconstruct loops from a control-flow graph.
    """

    loops: list[Loop] = []

    for back_edge in find_back_edges(cfg):
        header = back_edge.target
        source = back_edge.source

        body = _collect_loop_body(
            header,
            source,
            cfg,
        )

        loop_type = LoopType.UNKNOWN

        iterator = _find_instruction(
            header,
            _FOR_ITER_OPS,
        )

        if iterator is not None:
            loop_type = LoopType.FOR

        elif detect_while_loop(
            header,
            body,
        ):
            loop_type = LoopType.WHILE

        condition = None

        if loop_type == LoopType.WHILE:
            condition = _find_instruction(
                header,
                _CONDITIONAL_JUMPS,
            )

        loop = Loop(
            loop_type=loop_type,
            header=header,
            body=body,
            exit=_find_loop_exit(body),
            condition=condition,
            iterator=iterator,
        )

        loops.append(loop)

    return loops


# ---------------------------------------------------------------------------
# Break / continue detection
# ---------------------------------------------------------------------------


def find_continue_blocks(
    loop: Loop,
) -> list[BasicBlock]:
    """
    Find blocks that jump directly back to the loop header.

    These are strong candidates for `continue`.
    """

    result: list[BasicBlock] = []

    for block in loop.body:
        if block is loop.header:
            continue

        if loop.header in block.successors:
            result.append(block)

    return result


def find_break_blocks(
    loop: Loop,
) -> list[BasicBlock]:
    """
    Find blocks that leave the loop without being the normal loop exit.

    This is conservative because the later statement engine needs to
    inspect the exact branch condition before emitting `break`.
    """

    if loop.exit is None:
        return []

    body_ids = {
        id(block)
        for block in loop.body
    }

    result: list[BasicBlock] = []

    for block in loop.body:
        for successor in block.successors:
            if (
                successor is loop.exit
                and id(successor) not in body_ids
            ):
                result.append(block)

    return result


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def loop_type_name(
    loop: Loop,
) -> str:
    """Return a human-readable loop type."""

    return loop.loop_type.value


def describe_loop(
    loop: Loop,
) -> str:
    """Return a compact debugging representation."""

    header_bounds = _block_offsets(
        loop.header
    )

    if header_bounds is None:
        header = "unknown"
    else:
        header = str(
            header_bounds[0]
        )

    exit_offset = "none"

    if loop.exit is not None:
        bounds = _block_offsets(
            loop.exit
        )

        if bounds is not None:
            exit_offset = str(
                bounds[0]
            )

    return (
        f"{loop.loop_type.value} "
        f"header={header} "
        f"blocks={len(loop.body)} "
        f"exit={exit_offset}"
    )