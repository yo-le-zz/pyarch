from __future__ import annotations

"""
Structured control-flow reconstruction for PyArch.

Turns a flat CFG + linear per-block statements into nested
`If` / `While` / `For` IR nodes, instead of the flat, order-only
statement list produced by ``statements.py`` alone.

This pass is deliberately conservative: whenever a shape does not
match a pattern it understands, it falls back to the flat statement
list for the *whole* code object rather than emitting something that
looks structured but is wrong. A wrong "if" is worse than no "if".
"""

from .cfg import get_block
from .exception_table import get_exception_table
from .ir import (
    Break,
    Class,
    Continue,
    ControlFlowGraph,
    ExceptHandler,
    For,
    Function,
    If,
    IRStatement,
    Name,
    Pass,
    Raise,
    Try,
    TupleExpr,
    While,
    With,
    WithItem,
)
from .statements import (
    StatementError,
    reconstruct_statements,
)
from .stack import StackError
from .translate import TInstruction

_CONDITIONAL_OPS = frozenset(
    {
        "POP_JUMP_FORWARD_IF_FALSE",
        "POP_JUMP_FORWARD_IF_TRUE",
        "POP_JUMP_BACKWARD_IF_FALSE",
        "POP_JUMP_BACKWARD_IF_TRUE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
    }
)

_UNCONDITIONAL_OPS = frozenset(
    {
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP_ABSOLUTE",
    }
)

_TERMINATOR_OPS = frozenset(
    {
        "RETURN_VALUE",
        "RETURN_CONST",
        "RAISE_VARARGS",
        "RERAISE",
    }
)


class ControlFlowStructureError(Exception):
    """Raised when the CFG cannot be safely structured."""


# ---------------------------------------------------------------------------
# Block instruction lookup
# ---------------------------------------------------------------------------


def _split_instructions_by_block(
    instructions: list[TInstruction],
    cfg: ControlFlowGraph,
) -> dict[int, list[TInstruction]]:
    offset_to_block: dict[int, int] = {}

    for block in cfg.blocks:
        for ir_instruction in block.instructions:
            if ir_instruction.offset is not None:
                offset_to_block[ir_instruction.offset] = (
                    block.block_id
                )

    result: dict[int, list[TInstruction]] = {
        block.block_id: [] for block in cfg.blocks
    }

    for instruction in instructions:
        block_id = offset_to_block.get(instruction.offset)

        if block_id is not None:
            result[block_id].append(instruction)

    return result


def _block_for_offset(
    cfg: ControlFlowGraph,
    offset: int | None,
) -> int | None:
    if offset is None:
        return None

    for block in cfg.blocks:
        for ir_instruction in block.instructions:
            if ir_instruction.offset == offset:
                return block.block_id

    return None


def _fallthrough(
    cfg: ControlFlowGraph,
    block_id: int,
) -> int | None:
    """The block immediately following `block_id`, if any."""

    next_id = block_id + 1

    if get_block(cfg, next_id) is not None:
        return next_id

    return None


def _find_back_edge_source(
    cfg: ControlFlowGraph,
    header: int,
) -> int | None:
    """
    Return the furthest block that jumps back to `header`, if any.

    Used to detect that `header` is a loop header.
    """

    candidates = [
        block.block_id
        for block in cfg.blocks
        if block.block_id > header
        and header in block.successors
    ]

    return max(candidates) if candidates else None


def _is_conditional(instruction: TInstruction) -> bool:
    return instruction.op in _CONDITIONAL_OPS


def _is_none_return(statement: IRStatement) -> bool:
    from .ir import Return, Constant

    if not isinstance(statement, Return):
        return False

    value = getattr(statement, "value", None)

    return value is None or (
        isinstance(value, Constant) and value.value is None
    )


def _true_false_targets(
    cfg: ControlFlowGraph,
    header: int,
    instruction: TInstruction,
) -> tuple[int | None, int | None]:
    """
    Return (true_branch_block, false_branch_block) for a conditional
    jump, normalizing IF_TRUE / IF_FALSE variants.
    """

    target_id = _block_for_offset(
        cfg, instruction.target
    )
    fallthrough_id = _fallthrough(cfg, header)

    if "IF_FALSE" in instruction.op or instruction.op in (
        "JUMP_IF_FALSE_OR_POP",
    ):
        return fallthrough_id, target_id

    return target_id, fallthrough_id


# ---------------------------------------------------------------------------
# Region processing
# ---------------------------------------------------------------------------


def _region(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    start: int | None,
    stop: int | None,
    code_map: dict[int, "Function | Class"] | None,
    diagnostics,
    loop_stack: list[tuple[int, int | None]],
) -> tuple[list[IRStatement], int | None]:
    """
    Reconstruct statements for blocks from `start` up to (excluding)
    `stop`. Returns (statements, block_id_where_it_stopped).

    `block_id_where_it_stopped` is `stop` on a normal fallthrough
    exit, or None when the region ended in a return/break/continue
    (i.e. it never reaches `stop` through normal flow).
    """

    statements: list[IRStatement] = []
    current = start
    visited: set[int] = set()

    while current is not None and current != stop:
        if current in visited:
            diagnostics.warn(
                f"Control-flow cycle detected at block {current}; "
                "stopping structuring early."
            )
            return statements, None

        visited.add(current)

        block = get_block(cfg, current)

        if block is None:
            return statements, None

        instrs = instr_by_block.get(current, [])

        # ------------------------------------------------------------
        # with statement (must be checked before try/except: the
        # cleanup handler CPython generates for `with` also shows up
        # as a depth-0 exception-table entry, but needs different
        # handling)
        # ------------------------------------------------------------

        try_starts = getattr(
            diagnostics, "_pyarch_try_starts", None
        )

        with_index = None

        for index, instruction in enumerate(instrs):
            if instruction.op == "BEFORE_WITH":
                with_index = index
                break

        if with_index is not None:
            by_offset = getattr(
                diagnostics, "_pyarch_by_offset"
            )
            sorted_offsets = getattr(
                diagnostics, "_pyarch_sorted_offsets"
            )

            try:
                stmt, resume = _structure_with(
                    cfg,
                    instr_by_block,
                    by_offset,
                    sorted_offsets,
                    current,
                    instrs,
                    with_index,
                    try_starts,
                    code_map,
                    diagnostics,
                    loop_stack,
                )
            except ControlFlowStructureError as error:
                diagnostics.warn(
                    f"with-statement reconstruction failed, "
                    f"showing its body without the with wrapper: "
                    f"{error}"
                )
                body = _reconstruct(
                    instrs[:with_index], code_map, diagnostics
                )
                statements.extend(body)
                current = (
                    block.successors[0]
                    if block.successors
                    else None
                )
                continue

            if isinstance(stmt, list):
                statements.extend(stmt)
            else:
                statements.append(stmt)
            current = resume
            continue

        try_split_index = None
        try_entry = None

        if try_starts and instrs:
            for index, instruction in enumerate(instrs):
                if instruction.offset in try_starts:
                    try_split_index = index
                    try_entry = try_starts[instruction.offset]
                    break

        if try_entry is not None:
            # Consume it so the recursive _region call over this
            # same block (now starting at the try's own first
            # instruction) doesn't detect it again.
            del try_starts[try_entry.start]

            if try_split_index:
                # Statements before the try body (e.g. a module's
                # leading RESUME/NOP never produce one, but a real
                # statement could precede a try on the same block).
                statements.extend(
                    _reconstruct(
                        instrs[:try_split_index],
                        code_map,
                        diagnostics,
                    )
                )

            local_instr_by_block = dict(instr_by_block)
            local_instr_by_block[current] = instrs[
                try_split_index:
            ]

            by_offset = getattr(
                diagnostics, "_pyarch_by_offset"
            )
            sorted_offsets = getattr(
                diagnostics, "_pyarch_sorted_offsets"
            )

            try:
                stmt, resume = _structure_try(
                    cfg,
                    local_instr_by_block,
                    by_offset,
                    sorted_offsets,
                    try_entry,
                    code_map,
                    diagnostics,
                    loop_stack,
                )
            except ControlFlowStructureError as error:
                diagnostics.warn(
                    f"try/except/finally reconstruction failed, "
                    f"showing its body without the try wrapper: "
                    f"{error}"
                )
                # Safe degraded fallback: render the try body's own
                # statements without the wrapper, rather than
                # inventing a handler shape we're not sure about.
                body, _end = _region(
                    cfg,
                    local_instr_by_block,
                    current,
                    _block_for_offset(cfg, try_entry.target),
                    code_map,
                    diagnostics,
                    loop_stack,
                )
                statements.extend(body)
                current = _block_for_offset(
                    cfg, try_entry.target
                )
                continue

            statements.append(stmt)
            current = resume
            continue

        if not instrs:
            current = (
                block.successors[0]
                if block.successors
                else None
            )
            continue

        last = instrs[-1]

        # ------------------------------------------------------------
        # for loop (GET_ITER in this block, FOR_ITER alone in the
        # next -- the common CPython 3.11+ shape)
        # ------------------------------------------------------------

        if (
            last.op not in _UNCONDITIONAL_OPS
            and not _is_conditional(last)
            and len(block.successors) == 1
            and len(
                instr_by_block.get(
                    block.successors[0], []
                )
            )
            == 1
            and instr_by_block[block.successors[0]][0].op
            == "FOR_ITER"
        ):
            try:
                stack_result = _reconstruct_with_stack(
                    instrs, code_map
                )
                iterable = stack_result.stack.pop_expression()
            except (StatementError, StackError) as error:
                raise ControlFlowStructureError(
                    f"Could not evaluate for-loop iterable: {error}"
                ) from error

            statements.extend(stack_result.statements)

            stmt, resume = _structure_for(
                cfg,
                instr_by_block,
                block.successors[0],
                code_map,
                diagnostics,
                loop_stack,
                iterable=iterable,
            )
            statements.append(stmt)
            current = resume
            continue

        # ------------------------------------------------------------
        # for loop
        # ------------------------------------------------------------

        if last.op == "FOR_ITER":
            stmt, resume = _structure_for(
                cfg,
                instr_by_block,
                current,
                code_map,
                diagnostics,
                loop_stack,
            )
            statements.append(stmt)
            current = resume
            continue

        # ------------------------------------------------------------
        # while loop
        # ------------------------------------------------------------

        back_edge_source = _find_back_edge_source(
            cfg, current
        )

        if back_edge_source is not None and _is_conditional(
            last
        ):
            stmt, resume = _structure_while(
                cfg,
                instr_by_block,
                current,
                code_map,
                diagnostics,
                loop_stack,
            )
            statements.append(stmt)
            current = resume
            continue

        # ------------------------------------------------------------
        # if / else
        # ------------------------------------------------------------

        if _is_conditional(last):
            new_statements, resume = _structure_if(
                cfg,
                instr_by_block,
                current,
                code_map,
                diagnostics,
                loop_stack,
            )
            statements.extend(new_statements)
            current = resume
            continue

        # ------------------------------------------------------------
        # unconditional jump: break / continue / plain goto
        # ------------------------------------------------------------

        if last.op in _UNCONDITIONAL_OPS:
            body = _reconstruct(
                instrs[:-1], code_map, diagnostics
            )
            statements.extend(body)

            target_block = _block_for_offset(
                cfg, last.target
            )

            if loop_stack:
                loop_header, loop_exit = loop_stack[-1]

                if target_block == loop_exit:
                    statements.append(Break())
                    return statements, None

                if target_block == loop_header:
                    # Falling off the end of the body is
                    # semantically identical to `continue` here;
                    # only emit the keyword when it happens before
                    # the natural end of the region (i.e. jumping
                    # back from somewhere that is not simply "the
                    # last block of the loop").
                    if target_block != stop:
                        statements.append(Continue())
                    return statements, None

            # This jump leaves the current straight-line region
            # (e.g. the end of an `if` branch skipping past an
            # `else`). Stop here and let the caller compare the
            # target against what it expected, rather than walking
            # straight through a merge point and duplicating shared
            # tail code into this branch.
            return statements, target_block

        # ------------------------------------------------------------
        # plain fallthrough block (may end in return/raise)
        # ------------------------------------------------------------

        body = _reconstruct(instrs, code_map, diagnostics)

        if (
            loop_stack
            and body
            and _is_none_return(body[-1])
        ):
            # CPython sometimes compiles an early `break` as a direct
            # return when the loop is itself in tail position (the
            # code after the loop is just "the function/module
            # ends"). Recognize that shape here, with full loop
            # context available, rather than trying to distinguish
            # it from a genuine implicit return later.
            statements.extend(body[:-1])
            statements.append(Break())
            return statements, None

        statements.extend(body)

        if last.op in _TERMINATOR_OPS:
            return statements, None

        current = (
            block.successors[0]
            if block.successors
            else None
        )

    return statements, current


def _reconstruct(
    instructions: list[TInstruction],
    code_map: dict[int, "Function | Class"] | None,
    diagnostics,
) -> list[IRStatement]:
    if not instructions:
        return []

    try:
        result = reconstruct_statements(
            instructions, code_map
        )
    except StatementError as error:
        diagnostics.error(
            f"Statement reconstruction failed: {error}"
        )
        return []

    return result.statements


def _reconstruct_with_stack(
    instructions: list[TInstruction],
    code_map: dict[int, "Function | Class"] | None,
):
    """
    Like `reconstruct_statements`, but for the (four) call sites that
    also need the resulting stack to pop a trailing condition/
    iterable expression. Any failure becomes a
    `ControlFlowStructureError`, which `build_structured_body` always
    catches and turns into a safe flat fallback -- instead of an
    unhandled `StatementError`/`StackError` reaching the CLI.
    """

    try:
        return reconstruct_statements(instructions, code_map)
    except (StatementError, StackError) as error:
        raise ControlFlowStructureError(
            f"Could not reconstruct instructions: {error}"
        ) from error


# ---------------------------------------------------------------------------
# if / else
# ---------------------------------------------------------------------------


def _structure_if(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    header: int,
    code_map,
    diagnostics,
    loop_stack,
) -> tuple[list[IRStatement], int | None]:
    instrs = instr_by_block[header]
    condition_instr = instrs[-1]

    pre_statements = _reconstruct(
        instrs[:-1], code_map, diagnostics
    )

    stack_result = _reconstruct_with_stack(
        instrs[:-1], code_map
    )

    try:
        condition = stack_result.stack.pop_expression()
    except Exception as error:
        raise ControlFlowStructureError(
            f"Could not evaluate if-condition: {error}"
        ) from error

    true_id, false_id = _true_false_targets(
        cfg, header, condition_instr
    )

    then_statements, then_end = _region(
        cfg,
        instr_by_block,
        true_id,
        false_id,
        code_map,
        diagnostics,
        loop_stack,
    )

    if then_end == false_id:
        # Natural "no else" shape: the then-branch simply falls back
        # into whatever comes after the if.
        result = pre_statements + [
            If(
                test=condition,
                body=then_statements or [Pass()],
                orelse=[],
            )
        ]
        return result, false_id

    false_block = (
        get_block(cfg, false_id)
        if false_id is not None
        else None
    )

    false_is_dedicated = (
        false_block is not None
        and len(false_block.predecessors) == 1
        and header in false_block.predecessors
    )

    if not false_is_dedicated:
        # The then-branch terminated (return/break/...), and the
        # false-target is reachable from elsewhere too, so it is
        # shared code after the if, not an else-branch.
        result = pre_statements + [
            If(
                test=condition,
                body=then_statements or [Pass()],
                orelse=[],
            )
        ]
        return result, false_id

    else_statements, else_end = _region(
        cfg,
        instr_by_block,
        false_id,
        then_end,
        code_map,
        diagnostics,
        loop_stack,
    )

    merge_id = then_end if then_end is not None else else_end

    result = pre_statements + [
        If(
            test=condition,
            body=then_statements or [Pass()],
            orelse=else_statements,
        )
    ]
    return result, merge_id


# ---------------------------------------------------------------------------
# while
# ---------------------------------------------------------------------------


def _structure_while(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    header: int,
    code_map,
    diagnostics,
    loop_stack,
) -> tuple[IRStatement, int | None]:
    instrs = instr_by_block[header]
    condition_instr = instrs[-1]

    pre_statements = _reconstruct(
        instrs[:-1], code_map, diagnostics
    )

    stack_result = _reconstruct_with_stack(
        instrs[:-1], code_map
    )

    try:
        condition = stack_result.stack.pop_expression()
    except Exception as error:
        raise ControlFlowStructureError(
            f"Could not evaluate while-condition: {error}"
        ) from error

    body_start, exit_id = _true_false_targets(
        cfg, header, condition_instr
    )

    body_statements, _end = _region(
        cfg,
        instr_by_block,
        body_start,
        header,
        code_map,
        diagnostics,
        loop_stack + [(header, exit_id)],
    )

    # In CPython's usual "rotated" loop shape, the block that carries
    # the condition test is also the tail of the loop body (the body
    # falls straight back into its own condition check), so any
    # statements preceding the test in this block are genuine loop
    # body content, not code that runs once before the loop.
    node = While(
        test=condition,
        body=(pre_statements + body_statements) or [Pass()],
    )

    return node, exit_id


# ---------------------------------------------------------------------------
# for
# ---------------------------------------------------------------------------


def _extract_unpack_target_names(
    instrs: list[TInstruction],
) -> tuple[list[str] | None, int]:
    """
    Given a block starting with UNPACK_SEQUENCE/UNPACK_EX, return the
    target names in left-to-right order and how many instructions
    (the unpack plus whichever STORE instruction(s) follow -- one
    compound STORE_FAST_STORE_FAST, or several simple STOREs) were
    consumed. Returns (None, 0) if the shape isn't recognized, rather
    than guessing.
    """

    header = instrs[0]

    if header.op == "UNPACK_SEQUENCE":
        count = header.arg
    elif header.op == "UNPACK_EX":
        count = (
            (header.arg & 0xFF) + ((header.arg >> 8) & 0xFF) + 1
        )
    else:
        return None, 0

    if not isinstance(count, int):
        return None, 0

    names: list[str] = []
    index = 1

    while len(names) < count and index < len(instrs):
        instr = instrs[index]

        if instr.op == "STORE_FAST_STORE_FAST" and isinstance(
            instr.argval, tuple
        ):
            names.extend(instr.argval)
        elif instr.op in (
            "STORE_FAST",
            "STORE_NAME",
            "STORE_GLOBAL",
            "STORE_DEREF",
        ) and isinstance(instr.argval, str):
            names.append(instr.argval)
        else:
            return None, 0

        index += 1

    if len(names) != count:
        return None, 0

    return names, index


def _structure_for(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    header: int,
    code_map,
    diagnostics,
    loop_stack,
    iterable=None,
) -> tuple[IRStatement, int | None]:
    instrs = instr_by_block[header]
    for_iter_instr = instrs[-1]

    if iterable is None:
        stack_result = _reconstruct_with_stack(
            instrs[:-1], code_map
        )

        try:
            iterable = stack_result.stack.pop_expression()
        except Exception as error:
            raise ControlFlowStructureError(
                f"Could not evaluate for-loop iterable: {error}"
            ) from error

    body_start = _fallthrough(cfg, header)
    exit_id = _block_for_offset(
        cfg, for_iter_instr.target
    )

    target_expr: object = Name(name="_pyarch_unknown")

    if body_start is not None:
        body_instrs = instr_by_block.get(body_start, [])

        if body_instrs and body_instrs[0].op in {
            "STORE_FAST",
            "STORE_NAME",
            "STORE_GLOBAL",
            "STORE_DEREF",
        }:
            store_instr = body_instrs[0]

            if isinstance(store_instr.argval, str):
                target_expr = Name(name=store_instr.argval)
            elif isinstance(store_instr.argrepr, str):
                target_expr = Name(
                    name=store_instr.argrepr.strip()
                )

            # Replace this block's instructions (in our local copy of
            # the mapping) with everything after the STORE, so that
            # `_region` structures the rest of the block -- which may
            # itself end in a conditional jump (e.g. an `if` guarding
            # a `break`) -- instead of it being flattened.
            instr_by_block = dict(instr_by_block)
            instr_by_block[body_start] = body_instrs[1:]
        elif body_instrs and body_instrs[0].op in (
            "UNPACK_SEQUENCE",
            "UNPACK_EX",
        ):
            # `for a, b in pairs:` -- the loop variable is itself a
            # tuple-unpacking target.
            names, consumed = _extract_unpack_target_names(
                body_instrs
            )

            if names is not None:
                target_expr = (
                    TupleExpr(elements=[Name(name=n) for n in names])
                    if len(names) != 1
                    else Name(name=names[0])
                )

                instr_by_block = dict(instr_by_block)
                instr_by_block[body_start] = body_instrs[consumed:]
            else:
                diagnostics.warn(
                    "for-loop: could not determine the tuple-"
                    "unpacking loop variable; using a placeholder "
                    "name."
                )
        else:
            diagnostics.warn(
                "for-loop: could not determine the loop variable "
                "(possibly a tuple-unpacking target); using a "
                "placeholder name."
            )

    body_statements, _end = _region(
        cfg,
        instr_by_block,
        body_start,
        header,
        code_map,
        diagnostics,
        loop_stack + [(header, exit_id)],
    )

    node = For(
        target=target_expr,
        iter=iterable,
        body=body_statements or [Pass()],
    )

    return node, exit_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_structured_body(
    cfg: ControlFlowGraph | None,
    instructions: list[TInstruction],
    code_map: dict[int, "Function | Class"] | None,
    diagnostics,
    code=None,
) -> list[IRStatement]:
    """
    Build a structured statement tree (with nested If/While/For/Try)
    from a CFG, falling back to the flat statement list when
    structuring is not possible.

    `code` (a real CodeType or RemoteCode) is used to read the
    exception table for try/except/finally reconstruction; pass None
    to skip that (no exception table available or applicable).
    """

    if cfg is None or not cfg.blocks:
        return _reconstruct(instructions, code_map, diagnostics)

    instr_by_block = _split_instructions_by_block(
        instructions, cfg
    )

    by_offset, sorted_offsets = _build_flat_index(instructions)

    diagnostics._pyarch_by_offset = by_offset
    diagnostics._pyarch_sorted_offsets = sorted_offsets

    try_starts = {}

    if code is not None:
        for entry in get_exception_table(code):
            if entry.depth == 0:
                try_starts[entry.start] = entry
                continue

            # `with` statements produce a depth>0 entry (the extra
            # depth accounts for the __exit__ method CPython keeps
            # on the stack), identifiable by its handler starting
            # with PUSH_EXC_INFO immediately followed by
            # WITH_EXCEPT_START.
            first = by_offset.get(entry.target)
            second = (
                by_offset.get(
                    _next_offset(sorted_offsets, entry.target)
                )
                if first is not None
                else None
            )

            if (
                first is not None
                and first.op == "PUSH_EXC_INFO"
                and second is not None
                and second.op == "WITH_EXCEPT_START"
            ):
                try_starts[entry.start] = entry

    diagnostics._pyarch_try_starts = try_starts

    try:
        statements: list[IRStatement] = []
        current = cfg.entry
        seen_starts: set[int] = set()

        while current is not None:
            if current in seen_starts:
                diagnostics.warn(
                    f"Control-flow cycle detected while "
                    f"resuming at block {current}; stopping."
                )
                break

            seen_starts.add(current)

            part, current = _region(
                cfg,
                instr_by_block,
                current,
                None,
                code_map,
                diagnostics,
                [],
            )
            statements.extend(part)
    except ControlFlowStructureError as error:
        diagnostics.warn(
            f"Control-flow structuring failed, falling back to a "
            f"flat statement list: {error}"
        )
        return _reconstruct(instructions, code_map, diagnostics)

    return statements


def _is_empty_or_pass(body: list[IRStatement]) -> bool:
    return not body or all(
        isinstance(item, Pass) for item in body
    )


def _collapse_rotated_while(
    statements: list[IRStatement],
) -> list[IRStatement]:
    """
    CPython commonly compiles ``while cond: body`` by duplicating the
    condition test before the loop ("loop rotation"), which this pass
    reconstructs faithfully as::

        if cond:
            while cond:
                body

    That is correct but not idiomatic. When the outer `if`'s test is
    structurally identical to the inner `while`'s test and the `if`
    has no other content, collapse it back to a plain `while`.
    """

    result: list[IRStatement] = []

    for statement in statements:
        if isinstance(statement, If):
            statement.body = _collapse_rotated_while(
                statement.body
            )
            statement.orelse = _collapse_rotated_while(
                statement.orelse
            )

            if (
                len(statement.body) == 1
                and isinstance(statement.body[0], While)
                and statement.body[0].test == statement.test
                and _is_empty_or_pass(statement.orelse)
            ):
                result.append(statement.body[0])
                continue

        elif isinstance(statement, While):
            statement.body = _collapse_rotated_while(
                statement.body
            )
        elif isinstance(statement, For):
            statement.body = _collapse_rotated_while(
                statement.body
            )

        result.append(statement)

    return result


# ---------------------------------------------------------------------------
# try / except / finally
# ---------------------------------------------------------------------------


def _build_flat_index(
    instructions: list[TInstruction],
) -> tuple[dict[int, TInstruction], list[int]]:
    by_offset = {
        instruction.offset: instruction
        for instruction in instructions
    }
    return by_offset, sorted(by_offset)


def _next_offset(
    sorted_offsets: list[int],
    offset: int,
) -> int | None:
    import bisect

    index = bisect.bisect_right(sorted_offsets, offset)

    if index >= len(sorted_offsets):
        return None

    return sorted_offsets[index]


def _find_pop_except(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    start: int,
) -> int | None:
    offset = start

    while offset is not None:
        instr = by_offset.get(offset)

        if instr is None:
            return None

        if instr.op == "POP_EXCEPT":
            return offset

        offset = _next_offset(sorted_offsets, offset)

    return None


def _skip_as_cleanup(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    offset: int | None,
    name: str | None,
) -> int | None:
    """
    After `POP_EXCEPT`, CPython emits ``LOAD_CONST None; STORE_x
    name; DELETE_x name`` to unbind an ``as name`` target (PEP 3110).
    Skip that triple when present; it has no source-level
    representation. Anything else is left alone.
    """

    if name is None or offset is None:
        return offset

    first = by_offset.get(offset)

    if first is None or first.op != "LOAD_CONST":
        return offset

    second_offset = _next_offset(sorted_offsets, offset)
    second = by_offset.get(second_offset) if second_offset else None

    if second is None or second.op not in (
        "STORE_NAME",
        "STORE_FAST",
        "STORE_GLOBAL",
        "STORE_DEREF",
    ):
        return offset

    third_offset = _next_offset(sorted_offsets, second_offset)
    third = by_offset.get(third_offset) if third_offset else None

    if third is None or third.op not in (
        "DELETE_NAME",
        "DELETE_FAST",
        "DELETE_GLOBAL",
        "DELETE_DEREF",
    ):
        return offset

    return _next_offset(sorted_offsets, third_offset)


def _region_by_offset(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    start_offset: int,
    stop_offset: int | None,
    code_map,
    diagnostics,
    loop_stack,
) -> tuple[list[IRStatement], int | None]:
    """
    Like `_region`, but bounded by absolute instruction offsets
    instead of block ids -- needed because a try/except handler's
    body and its `POP_EXCEPT` often live in the very same CFG block
    (nothing forces a leader between them), so plain block-id bounds
    would treat the region as already empty.
    """

    start_block = _block_for_offset(cfg, start_offset)

    if start_block is None:
        return [], None

    local = dict(instr_by_block)
    local[start_block] = [
        instruction
        for instruction in local[start_block]
        if instruction.offset >= start_offset
    ]

    stop_block = (
        _block_for_offset(cfg, stop_offset)
        if stop_offset is not None
        else None
    )

    if stop_block == start_block:
        # Both ends fall inside the same block: trim its tail too and
        # process just that slice (its real successors are irrelevant
        # here -- we only want the statements strictly between the
        # two offsets).
        local[start_block] = [
            instruction
            for instruction in local[start_block]
            if stop_offset is None or instruction.offset < stop_offset
        ]

        return _region(
            cfg,
            local,
            start_block,
            None,
            code_map,
            diagnostics,
            loop_stack,
        )

    return _region(
        cfg,
        local,
        start_block,
        stop_block,
        code_map,
        diagnostics,
        loop_stack,
    )


def _structure_try(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    entry,
    code_map,
    diagnostics,
    loop_stack,
) -> tuple[IRStatement, int | None]:
    """
    Reconstruct one `try` statement from a depth-0 exception-table
    entry. Raises `ControlFlowStructureError` for any shape this
    does not confidently recognize (combined try/except/finally,
    try/else, or an unfamiliar handler pattern) rather than guessing
    -- the caller falls back to a flat, undecorated rendering of the
    same code, which stays correct even without the `try` wrapper.
    """

    try_start_block = _block_for_offset(cfg, entry.start)
    handler_start_block = _block_for_offset(cfg, entry.target)

    if try_start_block is None or handler_start_block is None:
        raise ControlFlowStructureError(
            "try: could not map exception-table offsets to blocks."
        )

    try_body_end = entry.end
    boundary_instr = by_offset.get(entry.end)

    if boundary_instr is not None and boundary_instr.op in (
        "RETURN_VALUE",
        "RETURN_CONST",
    ):
        # `return <expr>` inside a try: only the expression itself is
        # exception-protected, but the RETURN_* completing it belongs
        # to the same source statement and must be included, or the
        # expression is left dangling with nothing to consume it.
        try_body_end = _next_offset(sorted_offsets, entry.end)

    try_body, _try_end = _region_by_offset(
        cfg,
        instr_by_block,
        entry.start,
        try_body_end,
        code_map,
        diagnostics,
        loop_stack,
    )

    offset = entry.target
    first = by_offset.get(offset)

    if first is None or first.op != "PUSH_EXC_INFO":
        raise ControlFlowStructureError(
            "try: handler does not start with PUSH_EXC_INFO "
            "(unrecognized shape)."
        )

    offset = _next_offset(sorted_offsets, offset)

    # A plain `try/finally` (no `except`) has no CHECK_EXC_MATCH at
    # all: the handler just re-runs the finally body and re-raises.
    lookahead = by_offset.get(offset)

    if lookahead is not None and not _handler_chain_has_match(
        by_offset, sorted_offsets, offset
    ):
        finally_body, _end = _region(
            cfg,
            instr_by_block,
            _block_for_offset(cfg, offset),
            None,
            code_map,
            diagnostics,
            loop_stack,
        )

        # The `finally` body is duplicated by the compiler on the
        # success path too (right after the try body, before this
        # handler). Use the try body up to that duplication point:
        # `_try_end` already stopped there, so `try_body` is correct
        # as-is; the handler copy (which always ends in RERAISE) is
        # only needed to recover the finally body's statements.
        finally_body = [
            statement
            for statement in finally_body
            if not isinstance(statement, Raise)
        ]

        return (
            Try(
                body=try_body or [Pass()],
                handlers=[],
                orelse=[],
                finalbody=finally_body or [Pass()],
            ),
            None,
        )

    handlers: list[ExceptHandler] = []
    resumes: list[int | None] = []
    saw_bare = False

    while True:
        instr = by_offset.get(offset)

        if instr is None:
            raise ControlFlowStructureError(
                "try: ran out of instructions while parsing handlers."
            )

        if instr.op in ("RERAISE",):
            break

        if saw_bare:
            raise ControlFlowStructureError(
                "try: unexpected clause after a bare 'except:'."
            )

        exc_type_expr = None
        next_check_target = None

        if instr.op == "POP_TOP":
            saw_bare = True
            offset = _next_offset(sorted_offsets, offset)
        elif instr.op == "CHECK_EXC_MATCH":
            # LOAD-only type expression already consumed below in
            # the general case; a bare CHECK_EXC_MATCH right away
            # means the type is a single preceding LOAD we haven't
            # collected -- treat as malformed and bail out.
            raise ControlFlowStructureError(
                "try: exception type expression missing."
            )
        else:
            type_instrs: list[TInstruction] = []

            while instr is not None and instr.op != "CHECK_EXC_MATCH":
                type_instrs.append(instr)
                offset = _next_offset(sorted_offsets, offset)
                instr = by_offset.get(offset)

            if instr is None:
                raise ControlFlowStructureError(
                    "try: malformed exception-type check."
                )

            offset = _next_offset(sorted_offsets, offset)
            jump_instr = by_offset.get(offset)

            if jump_instr is None or "IF_FALSE" not in jump_instr.op:
                raise ControlFlowStructureError(
                    "try: expected a jump after CHECK_EXC_MATCH."
                )

            next_check_target = jump_instr.target
            offset = _next_offset(sorted_offsets, offset)

            type_result = _reconstruct_with_stack(
                type_instrs, code_map
            )

            try:
                exc_type_expr = type_result.stack.pop_expression()
            except Exception as error:
                raise ControlFlowStructureError(
                    f"try: could not evaluate exception type: {error}"
                ) from error

            instr = by_offset.get(offset)

        as_name: str | None = None

        if instr is not None and instr.op in (
            "STORE_NAME",
            "STORE_FAST",
            "STORE_GLOBAL",
            "STORE_DEREF",
        ):
            as_name = (
                instr.argval
                if isinstance(instr.argval, str)
                else None
            )
            offset = _next_offset(sorted_offsets, offset)
        elif instr is not None and instr.op == "POP_TOP":
            offset = _next_offset(sorted_offsets, offset)
        else:
            raise ControlFlowStructureError(
                "try: unexpected instruction after exception match."
            )

        body_start_offset = offset
        pop_except_offset = _find_pop_except(
            by_offset, sorted_offsets, body_start_offset
        )

        if pop_except_offset is None:
            raise ControlFlowStructureError(
                "try: could not find POP_EXCEPT for a handler."
            )

        after_cleanup_offset = _skip_as_cleanup(
            by_offset,
            sorted_offsets,
            _next_offset(sorted_offsets, pop_except_offset),
            as_name,
        )

        after_cleanup_instr = (
            by_offset.get(after_cleanup_offset)
            if after_cleanup_offset is not None
            else None
        )

        if after_cleanup_instr is not None and (
            after_cleanup_instr.op
            in ("RETURN_VALUE", "RETURN_CONST")
        ):
            # The handler ends in `return <expr>`, where <expr>'s
            # value was protected across POP_EXCEPT by a SWAP this
            # model doesn't track: include everything through the
            # RETURN in one pass so that value gets consumed into a
            # real Return statement instead of being left dangling.
            body_end_offset = _next_offset(
                sorted_offsets, after_cleanup_offset
            )
        else:
            body_end_offset = pop_except_offset

        handler_body, handler_end = _region_by_offset(
            cfg,
            instr_by_block,
            body_start_offset,
            body_end_offset,
            code_map,
            diagnostics,
            loop_stack,
        )

        handler_body = _strip_as_cleanup_statements(
            handler_body, as_name
        )

        offset = _next_offset(sorted_offsets, pop_except_offset)
        offset = _skip_as_cleanup(
            by_offset, sorted_offsets, offset, as_name
        )

        resume_after_handler = None
        resume_instr = by_offset.get(offset) if offset else None

        if resume_instr is not None and resume_instr.op not in (
            "RETURN_VALUE",
            "RETURN_CONST",
            "RAISE_VARARGS",
            "RERAISE",
        ):
            resume_after_handler = _block_for_offset(cfg, offset)

        handlers.append(
            ExceptHandler(
                type=exc_type_expr,
                name=as_name,
                body=handler_body or [Pass()],
            )
        )
        resumes.append(resume_after_handler)

        if saw_bare or next_check_target is None:
            break

        offset = next_check_target

    resume = next(
        (r for r in resumes if r is not None),
        None,
    )

    return (
        Try(
            body=try_body or [Pass()],
            handlers=handlers,
            orelse=[],
            finalbody=[],
        ),
        resume,
    )


def _strip_as_cleanup_statements(
    statements: list[IRStatement],
    as_name: str | None,
) -> list[IRStatement]:
    """
    Remove a PEP 3110 ``as``-name cleanup (``name = None; del name``)
    if it was reconstructed as real statements (this happens when the
    handler body's end boundary was extended to include a trailing
    `return`, which pulls POP_EXCEPT and the cleanup triple into the
    same reconstructed range).
    """

    if as_name is None:
        return statements

    from .ir import Assign, Delete, Constant, Name

    result = list(statements)

    for index in range(len(result) - 1):
        first, second = result[index], result[index + 1]

        if (
            isinstance(first, Assign)
            and isinstance(first.target, Name)
            and first.target.name == as_name
            and isinstance(first.value, Constant)
            and first.value.value is None
            and isinstance(second, Delete)
            and isinstance(second.target, Name)
            and second.target.name == as_name
        ):
            del result[index : index + 2]
            break

    return result


def _handler_chain_has_match(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    start: int,
) -> bool:
    """
    True if this is a real `except` chain (bare or typed) rather than
    a plain `finally` (no `except` at all).

    A `finally`-only handler never installs a match: it just re-runs
    the finally body and unconditionally re-raises, so it never
    reaches a `POP_EXCEPT`. Every `except` clause -- even a bare
    ``except:`` -- always runs `POP_EXCEPT` on its normal path, since
    catching the exception necessarily uninstalls it. This is more
    reliable than checking for `CHECK_EXC_MATCH`, which a bare
    `except:` doesn't have either.
    """

    offset = start

    while offset is not None:
        instr = by_offset.get(offset)

        if instr is None:
            return False

        if instr.op == "POP_EXCEPT":
            return True

        if instr.op == "RERAISE":
            return False

        offset = _next_offset(sorted_offsets, offset)

    return False


def _skip_with_cleanup(
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    start: int,
) -> int | None:
    """
    Skip CPython's normal-path `__exit__(None, None, None)` cleanup
    call that follows a `with` body -- pure compiler machinery with
    no source-level representation. Tolerant of minor shape
    variation: skips any leading LOAD_CONST/PUSH_NULL setup, then one
    CALL, then the POP_TOP that discards its result.
    """

    offset = start

    while True:
        instr = by_offset.get(offset)

        if instr is None:
            return None

        if instr.op == "CALL":
            offset = _next_offset(sorted_offsets, offset)
            break

        if instr.op in ("LOAD_CONST", "PUSH_NULL"):
            offset = _next_offset(sorted_offsets, offset)
            continue

        return None

    instr = by_offset.get(offset)

    if instr is not None and instr.op == "POP_TOP":
        return _next_offset(sorted_offsets, offset)

    return offset


def _structure_with(
    cfg: ControlFlowGraph,
    instr_by_block: dict[int, list[TInstruction]],
    by_offset: dict[int, TInstruction],
    sorted_offsets: list[int],
    current: int,
    instrs: list[TInstruction],
    with_index: int,
    try_starts: dict | None,
    code_map,
    diagnostics,
    loop_stack,
) -> tuple[IRStatement, int | None]:
    ctx_expr = None

    try:
        pre_result = _reconstruct_with_stack(
            instrs[:with_index], code_map
        )
        ctx_expr = pre_result.stack.pop_expression()
    except Exception as error:
        raise ControlFlowStructureError(
            f"with: could not evaluate the context expression: "
            f"{error}"
        ) from error

    pre_statements = pre_result.statements

    before_with_offset = instrs[with_index].offset
    after_offset = _next_offset(sorted_offsets, before_with_offset)

    entry = (try_starts or {}).pop(after_offset, None)

    if entry is None:
        raise ControlFlowStructureError(
            "with: no matching exception-table entry found "
            "(unrecognized shape)."
        )

    bind_instr = by_offset.get(after_offset)
    as_name = None

    if bind_instr is not None and bind_instr.op in (
        "STORE_FAST",
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_DEREF",
    ) and isinstance(bind_instr.argval, str):
        as_name = bind_instr.argval
        body_start_offset = _next_offset(sorted_offsets, after_offset)
    elif bind_instr is not None and bind_instr.op == "POP_TOP":
        body_start_offset = _next_offset(sorted_offsets, after_offset)
    else:
        raise ControlFlowStructureError(
            "with: unexpected instruction binding the context "
            "manager result."
        )

    body, _end = _region_by_offset(
        cfg,
        instr_by_block,
        body_start_offset,
        entry.end,
        code_map,
        diagnostics,
        loop_stack,
    )

    resume = _skip_with_cleanup(
        by_offset, sorted_offsets, entry.end
    )

    node = With(
        items=[
            WithItem(
                context_expr=ctx_expr,
                optional_vars=(
                    Name(name=as_name)
                    if as_name is not None
                    else None
                ),
            )
        ],
        body=body or [Pass()],
    )

    if pre_statements:
        return [*pre_statements, node], resume

    return node, resume


__all__ = [
    "ControlFlowStructureError",
    "build_structured_body",
    "collapse_rotated_while",
]

# Public alias (the leading underscore is an implementation detail).
collapse_rotated_while = _collapse_rotated_while
