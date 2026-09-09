from __future__ import annotations

"""
Exception reconstruction for PyArch.

This module detects exception-related bytecode instructions and provides
a version-independent representation for the decompiler engine.

The actual conversion to Python AST will happen later in engine.py.
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


class ExceptionError(Exception):
    """Raised when exception reconstruction fails."""


# ---------------------------------------------------------------------------
# Exception regions
# ---------------------------------------------------------------------------


class ExceptionRegionType(str, Enum):
    """Types of Python exception-handling regions."""

    TRY = "try"
    EXCEPT = "except"
    FINALLY = "finally"
    EXCEPT_STAR = "except*"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ExceptionHandler:
    """
    A reconstructed exception handler.

    `handler` points to the bytecode block containing the exception
    handling code.

    `exception_type` is the expression used by an `except` clause when
    it can be recovered.
    """

    handler: BasicBlock

    exception_type: IRInstruction | None = None

    variable: str | None = None

    is_star: bool = False


@dataclass(slots=True)
class ExceptionRegion:
    """
    A reconstructed exception-handling region.
    """

    region_type: ExceptionRegionType

    start: BasicBlock

    end: BasicBlock | None = None

    handlers: list[ExceptionHandler] | None = None

    finally_block: BasicBlock | None = None

    body: list[BasicBlock] | None = None

    def __post_init__(self) -> None:
        if self.handlers is None:
            self.handlers = []

        if self.body is None:
            self.body = []


# ---------------------------------------------------------------------------
# Opcode classification
# ---------------------------------------------------------------------------


_SETUP_EXCEPT = {
    "SETUP_EXCEPT",
    "SETUP_FINALLY",
    "SETUP_WITH",
    "SETUP_ASYNC_WITH",
}

_EXCEPTION_OPS = {
    "CHECK_EXC_MATCH",
    "CHECK_EG_MATCH",
    "PUSH_EXC_INFO",
    "POP_EXCEPT",
    "RERAISE",
    "PREP_RERAISE_STAR",
    "WITH_EXCEPT_START",
}

_RAISE_OPS = {
    "RAISE_VARARGS",
    "RAISE",
}

_FINALLY_OPS = {
    "RERAISE",
    "END_FINALLY",
}


def is_exception_instruction(
    instruction: IRInstruction,
) -> bool:
    """Return whether an instruction is exception-related."""

    return (
        instruction.op in _EXCEPTION_OPS
        or instruction.op in _SETUP_EXCEPT
        or instruction.op in _RAISE_OPS
        or instruction.op in _FINALLY_OPS
    )


def is_raise_instruction(
    instruction: IRInstruction,
) -> bool:
    """Return whether an instruction explicitly raises an exception."""

    return instruction.op in _RAISE_OPS


def is_finally_instruction(
    instruction: IRInstruction,
) -> bool:
    """Return whether an instruction belongs to finally machinery."""

    return instruction.op in _FINALLY_OPS


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------


def _block_start(
    block: BasicBlock,
) -> int | None:
    """Return the first bytecode offset of a block."""

    if not block.instructions:
        return None

    return block.instructions[0].offset


def _block_end(
    block: BasicBlock,
) -> int | None:
    """Return the final bytecode offset of a block."""

    if not block.instructions:
        return None

    return block.instructions[-1].offset


def _contains_exception_instruction(
    block: BasicBlock,
) -> bool:
    """Return whether a block contains exception machinery."""

    return any(
        is_exception_instruction(
            instruction
        )
        for instruction in block.instructions
    )


def _find_instruction(
    block: BasicBlock,
    operations: set[str],
) -> IRInstruction | None:
    """Find the first matching instruction in a block."""

    for instruction in block.instructions:
        if instruction.op in operations:
            return instruction

    return None


# ---------------------------------------------------------------------------
# Raise detection
# ---------------------------------------------------------------------------


def find_raise_instructions(
    cfg: ControlFlowGraph,
) -> list[tuple[BasicBlock, IRInstruction]]:
    """
    Find explicit raise instructions in a CFG.
    """

    result: list[
        tuple[BasicBlock, IRInstruction]
    ] = []

    for block in cfg.blocks:
        for instruction in block.instructions:
            if is_raise_instruction(
                instruction
            ):
                result.append(
                    (
                        block,
                        instruction,
                    )
                )

    return result


# ---------------------------------------------------------------------------
# Handler detection
# ---------------------------------------------------------------------------


def is_exception_handler(
    block: BasicBlock,
) -> bool:
    """
    Detect whether a block looks like an exception handler.

    Modern CPython bytecode commonly starts exception handlers with
    PUSH_EXC_INFO / CHECK_EXC_MATCH or related instructions.
    """

    if not block.instructions:
        return False

    first = block.instructions[0]

    if first.op in {
        "PUSH_EXC_INFO",
        "CHECK_EXC_MATCH",
        "CHECK_EG_MATCH",
        "WITH_EXCEPT_START",
    }:
        return True

    return _contains_exception_instruction(
        block
    )


def find_exception_handlers(
    cfg: ControlFlowGraph,
) -> list[BasicBlock]:
    """Return blocks that appear to contain exception handlers."""

    return [
        block
        for block in cfg.blocks
        if is_exception_handler(block)
    ]


# ---------------------------------------------------------------------------
# Finally detection
# ---------------------------------------------------------------------------


def is_finally_block(
    block: BasicBlock,
) -> bool:
    """Return whether a block appears to implement finally logic."""

    return any(
        is_finally_instruction(
            instruction
        )
        for instruction in block.instructions
    )


def find_finally_blocks(
    cfg: ControlFlowGraph,
) -> list[BasicBlock]:
    """Return blocks containing finally machinery."""

    return [
        block
        for block in cfg.blocks
        if is_finally_block(block)
    ]


# ---------------------------------------------------------------------------
# Exception region discovery
# ---------------------------------------------------------------------------


def _reachable_blocks(
    start: BasicBlock,
    *,
    stop: set[int] | None = None,
) -> list[BasicBlock]:
    """
    Walk blocks reachable from `start`.

    This is intentionally local and conservative. The engine will later
    use CFG dominance/post-dominance information for more precise
    region reconstruction.
    """

    if stop is None:
        stop = set()

    visited: set[int] = set()
    result: list[BasicBlock] = []

    worklist = [start]

    while worklist:
        block = worklist.pop()

        identity = id(block)

        if identity in visited:
            continue

        if identity in stop:
            continue

        visited.add(identity)
        result.append(block)

        for successor in block.successors:
            if id(successor) not in visited:
                worklist.append(
                    successor
                )

    return result


def detect_exception_region(
    cfg: ControlFlowGraph,
    start: BasicBlock,
) -> ExceptionRegion | None:
    """
    Attempt to identify an exception region beginning at `start`.

    This function deliberately avoids assuming one CPython bytecode
    layout. Python 3.8 through 3.13 use significantly different
    exception-handling bytecode.
    """

    reachable = _reachable_blocks(
        start
    )

    handlers = [
        block
        for block in reachable
        if is_exception_handler(block)
    ]

    finally_blocks = [
        block
        for block in reachable
        if is_finally_block(block)
    ]

    if not handlers and not finally_blocks:
        return None

    if finally_blocks:
        region_type = (
            ExceptionRegionType.FINALLY
        )
    else:
        region_type = (
            ExceptionRegionType.EXCEPT
        )

    exception_handlers = [
        ExceptionHandler(
            handler=block,
            is_star=any(
                instruction.op
                == "CHECK_EG_MATCH"
                for instruction
                in block.instructions
            ),
        )
        for block in handlers
    ]

    body = [
        block
        for block in reachable
        if block not in handlers
        and block not in finally_blocks
    ]

    return ExceptionRegion(
        region_type=region_type,
        start=start,
        handlers=exception_handlers,
        finally_block=(
            finally_blocks[0]
            if finally_blocks
            else None
        ),
        body=body,
    )


def reconstruct_exceptions(
    cfg: ControlFlowGraph,
) -> list[ExceptionRegion]:
    """
    Discover exception regions in a CFG.
    """

    regions: list[ExceptionRegion] = []

    candidates = [
        block
        for block in cfg.blocks
        if _contains_exception_instruction(
            block
        )
    ]

    seen: set[
        tuple[int, ExceptionRegionType]
    ] = set()

    for block in candidates:
        region = detect_exception_region(
            cfg,
            block,
        )

        if region is None:
            continue

        key = (
            id(region.start),
            region.region_type,
        )

        if key in seen:
            continue

        seen.add(key)
        regions.append(region)

    return regions


# ---------------------------------------------------------------------------
# Rendering / debugging helpers
# ---------------------------------------------------------------------------


def describe_handler(
    handler: ExceptionHandler,
) -> str:
    """Return a compact handler description."""

    offset = _block_start(
        handler.handler
    )

    if offset is None:
        offset_text = "unknown"
    else:
        offset_text = str(offset)

    suffix = (
        " except*"
        if handler.is_star
        else " except"
    )

    if handler.variable is not None:
        suffix += (
            f" as {handler.variable}"
        )

    return (
        f"handler@{offset_text}"
        f"{suffix}"
    )


def describe_region(
    region: ExceptionRegion,
) -> str:
    """Return a compact exception-region description."""

    start = _block_start(
        region.start
    )

    end = (
        _block_end(region.end)
        if region.end is not None
        else None
    )

    handlers = len(
        region.handlers or []
    )

    return (
        f"{region.region_type.value} "
        f"start={start} "
        f"end={end} "
        f"handlers={handlers}"
    )


def render_exception_debug(
    regions: list[ExceptionRegion],
) -> str:
    """
    Render exception regions for debugging.

    This is not final Python source generation. The final writer will
    use Python's `ast` module.
    """

    lines: list[str] = []

    for region in regions:
        lines.append(
            describe_region(region)
        )

        for handler in (
            region.handlers or []
        ):
            lines.append(
                f"    "
                f"{describe_handler(handler)}"
            )

        if region.finally_block is not None:
            offset = _block_start(
                region.finally_block
            )

            lines.append(
                "    "
                f"finally@{offset}"
            )

    return "\n".join(lines)