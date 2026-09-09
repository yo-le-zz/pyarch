from __future__ import annotations

"""
Control-flow graph construction for PyArch.

Pipeline:

    TInstruction
         ↓
    basic blocks
         ↓
    ControlFlowGraph
         ↓
    structured control flow
         ↓
    Python statements
"""


from dataclasses import dataclass

from .ir import (
    BasicBlock,
    ControlFlowGraph,
    IRInstruction,
)
from .translate import TInstruction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CONDITIONAL_JUMPS = frozenset(
    {
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
    }
)


_UNCONDITIONAL_JUMPS = frozenset(
    {
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP_ABSOLUTE",
    }
)


_TERMINATORS = frozenset(
    {
        "RETURN_VALUE",
        "RETURN_CONST",
        "RAISE_VARARGS",
        "RERAISE",
    }
)


@dataclass(slots=True, frozen=True)
class CFGEdge:
    """A directed edge between two basic blocks."""

    source: int
    target: int


# ---------------------------------------------------------------------------
# Instruction conversion
# ---------------------------------------------------------------------------


def _to_ir_instruction(
    instruction: TInstruction,
) -> IRInstruction:
    """
    Convert a normalized translated instruction into the generic IR form.
    """

    return IRInstruction(
        offset=instruction.offset,
        op=instruction.op,
        arg=instruction.arg,
        value=instruction.argval,
        target=instruction.target,
    )


# ---------------------------------------------------------------------------
# Leaders
# ---------------------------------------------------------------------------


def _instruction_offsets(
    instructions: list[TInstruction],
) -> set[int]:
    """Return every instruction offset."""

    return {
        instruction.offset
        for instruction in instructions
    }


def _next_offset(
    instructions: list[TInstruction],
    index: int,
) -> int | None:
    """Return the offset of the instruction following `index`."""

    next_index = index + 1

    if next_index >= len(instructions):
        return None

    return instructions[next_index].offset


def find_leaders(
    instructions: list[TInstruction],
) -> list[int]:
    """
    Find basic-block leaders.

    A leader is:

    - the first instruction;
    - a jump target;
    - an instruction immediately following a jump;
    - an instruction immediately following a terminator.
    """

    if not instructions:
        return []

    offsets = _instruction_offsets(
        instructions
    )

    leaders: set[int] = {
        instructions[0].offset
    }

    for index, instruction in enumerate(
        instructions
    ):
        op = instruction.op

        # Any valid jump target starts a new block.
        if instruction.target is not None:
            if instruction.target in offsets:
                leaders.add(
                    instruction.target
                )

        # Conditional and unconditional jumps
        # terminate the current block.
        if (
            op in _CONDITIONAL_JUMPS
            or op in _UNCONDITIONAL_JUMPS
        ):
            next_offset = _next_offset(
                instructions,
                index,
            )

            if next_offset is not None:
                leaders.add(
                    next_offset
                )

        # Returns and exceptions terminate a block.
        elif op in _TERMINATORS:
            next_offset = _next_offset(
                instructions,
                index,
            )

            if next_offset is not None:
                leaders.add(
                    next_offset
                )

    return sorted(leaders)


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


def _split_into_blocks(
    instructions: list[TInstruction],
    leaders: list[int],
) -> list[BasicBlock]:
    """Split instructions into basic blocks."""

    if not instructions or not leaders:
        return []

    leader_set = set(leaders)

    blocks: list[BasicBlock] = []

    current: BasicBlock | None = None

    for instruction in instructions:
        if instruction.offset in leader_set:
            current = BasicBlock(
                block_id=len(blocks),
                instructions=[],
            )

            blocks.append(
                current
            )

        if current is None:
            raise RuntimeError(
                "CFG construction encountered "
                "an instruction before the first block."
            )

        current.instructions.append(
            _to_ir_instruction(
                instruction
            )
        )

    return blocks


# ---------------------------------------------------------------------------
# Block lookup
# ---------------------------------------------------------------------------


def _build_offset_map(
    blocks: list[BasicBlock],
) -> dict[int, int]:
    """
    Map every instruction offset to its block ID.
    """

    mapping: dict[int, int] = {}

    for block in blocks:
        for instruction in block.instructions:
            if instruction.offset is None:
                continue

            mapping[instruction.offset] = (
                block.block_id
            )

    return mapping


def _block_for_offset(
    blocks: list[BasicBlock],
    offset: int,
) -> int | None:
    """Return the block containing an instruction offset."""

    for block in blocks:
        for instruction in block.instructions:
            if instruction.offset == offset:
                return block.block_id

    return None


# ---------------------------------------------------------------------------
# Successors
# ---------------------------------------------------------------------------


def _last_instruction(
    block: BasicBlock,
) -> IRInstruction | None:
    """Return the final instruction in a block."""

    if not block.instructions:
        return None

    return block.instructions[-1]


def _add_successor(
    block: BasicBlock,
    target: int,
) -> None:
    """Add a successor without duplicates."""

    if target not in block.successors:
        block.successors.append(
            target
        )


def _build_successors(
    blocks: list[BasicBlock],
    instructions: list[TInstruction],
) -> None:
    """Build successor edges for every basic block."""

    offset_to_block = _build_offset_map(
        blocks
    )

    for index, block in enumerate(blocks):
        last = _last_instruction(block)

        if last is None:
            continue

        op = last.op

        # ---------------------------------------------------------------
        # Conditional jump
        # ---------------------------------------------------------------

        if op in _CONDITIONAL_JUMPS:
            if last.target is not None:
                target_block = offset_to_block.get(
                    last.target
                )

                if target_block is not None:
                    _add_successor(
                        block,
                        target_block,
                    )

            # Conditional jumps also fall through.
            if index + 1 < len(blocks):
                _add_successor(
                    block,
                    blocks[index + 1].block_id,
                )

            continue

        # ---------------------------------------------------------------
        # Unconditional jump
        # ---------------------------------------------------------------

        if op in _UNCONDITIONAL_JUMPS:
            if last.target is not None:
                target_block = offset_to_block.get(
                    last.target
                )

                if target_block is not None:
                    _add_successor(
                        block,
                        target_block,
                    )

            continue

        # ---------------------------------------------------------------
        # Terminator
        # ---------------------------------------------------------------

        if op in _TERMINATORS:
            continue

        # ---------------------------------------------------------------
        # Normal fall-through
        # ---------------------------------------------------------------

        if index + 1 < len(blocks):
            _add_successor(
                block,
                blocks[index + 1].block_id,
            )


# ---------------------------------------------------------------------------
# Predecessors
# ---------------------------------------------------------------------------


def _build_predecessors(
    blocks: list[BasicBlock],
) -> None:
    """Build predecessor lists from successor lists."""

    for block in blocks:
        block.predecessors.clear()

    for block in blocks:
        for successor in block.successors:
            if successor < 0:
                continue

            if successor >= len(blocks):
                continue

            target = blocks[successor]

            if block.block_id not in target.predecessors:
                target.predecessors.append(
                    block.block_id
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cfg(
    instructions: list[TInstruction],
) -> ControlFlowGraph:
    """
    Build a control-flow graph from translated instructions.
    """

    if not instructions:
        return ControlFlowGraph(
            blocks=[],
            entry=None,
        )

    leaders = find_leaders(
        instructions
    )

    blocks = _split_into_blocks(
        instructions,
        leaders,
    )

    _build_successors(
        blocks,
        instructions,
    )

    _build_predecessors(
        blocks
    )

    return ControlFlowGraph(
        blocks=blocks,
        entry=(
            blocks[0].block_id
            if blocks
            else None
        ),
    )


def iter_edges(
    cfg: ControlFlowGraph,
):
    """Yield every CFG edge."""

    for block in cfg.blocks:
        for successor in block.successors:
            yield CFGEdge(
                source=block.block_id,
                target=successor,
            )


def get_block(
    cfg: ControlFlowGraph,
    block_id: int,
) -> BasicBlock | None:
    """Return a block by ID."""

    if block_id < 0:
        return None

    if block_id >= len(cfg.blocks):
        return None

    return cfg.blocks[block_id]


def block_for_offset(
    cfg: ControlFlowGraph,
    offset: int,
) -> BasicBlock | None:
    """Return the basic block containing an instruction offset."""

    for block in cfg.blocks:
        for instruction in block.instructions:
            if instruction.offset == offset:
                return block

    return None


def validate_cfg(
    cfg: ControlFlowGraph,
) -> list[str]:
    """
    Validate the basic structural integrity of a CFG.

    Returns a list of errors instead of raising immediately, which lets
    PyArch report partially recovered bytecode gracefully.
    """

    errors: list[str] = []

    if cfg.entry is not None:
        if cfg.entry < 0 or cfg.entry >= len(cfg.blocks):
            errors.append(
                f"Invalid entry block: {cfg.entry}"
            )

    known_ids = {
        block.block_id
        for block in cfg.blocks
    }

    for block in cfg.blocks:
        if block.block_id not in known_ids:
            errors.append(
                f"Unknown block ID: {block.block_id}"
            )

        for successor in block.successors:
            if successor not in known_ids:
                errors.append(
                    f"Block {block.block_id} "
                    f"has invalid successor {successor}"
                )

        for predecessor in block.predecessors:
            if predecessor not in known_ids:
                errors.append(
                    f"Block {block.block_id} "
                    f"has invalid predecessor {predecessor}"
                )

    return errors