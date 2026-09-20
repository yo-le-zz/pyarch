# PyArch

A Python binary reconstruction and analysis tool: extracts PyInstaller
executables, decompiles their `.pyc` bytecode back into readable
Python source, reconstructs a project layout, and reports what it
found (and what it couldn't).

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10 to *run* PyArch. See "Cross-version bytecode"
below for what PyArch can *analyze*.

## CLI

```bash
pyarch extract program.exe        # unpack a PyInstaller executable
pyarch decompile <extracted-dir>  # .pyc -> .py (application code only)
pyarch decompile <dir> --all      # include bundled stdlib/third-party code
pyarch reconstruct <extracted-dir> # build a project layout (pyproject.toml, packages)
pyarch inspect <extracted-dir>    # JSON report: what was recovered, warnings, errors
pyarch make program.exe           # extract -> decompile -> reconstruct -> inspect
pyarch --version
pyarch --help
```

## Architecture

```
CArchive / PYZ extraction (extractor.py, carchive.py, pyz.py, formats/)
        v
.pyc parsing + cross-version dispatch (pyc.py, versions.py, coderef.py)
        v
bytecode -> normalized instructions (decompiler/translate.py)
        v
control-flow graph (decompiler/cfg.py)
        v
structured reconstruction: if/while/for/try, functions, classes,
imports (decompiler/control_flow.py, statements.py, stack.py,
functions.py, classes.py, exception_table.py)
        v
IR (decompiler/ir.py) -> Python ast -> ast.unparse() (decompiler/writer.py)
        v
.py source, validated with ast.parse() + compile()
        v
project reconstruction (reconstructor.py) + inspection report (inspector.py)
```

The IR stays the boundary between CPython bytecode and Python's own
`ast` module; nothing downstream of `translate.py` looks at raw
opcodes.

## Cross-version bytecode

A `.pyc`'s magic number is checked against the Python version running
PyArch (`versions.py`). When they match, PyArch disassembles it
directly. When they don't -- e.g. analyzing a Python 3.13 executable
while running PyArch under 3.12 -- disassembling in-process would
silently use the wrong opcode table and produce incorrect
instructions with no error at all.

Instead, PyArch looks for a matching interpreter on `PATH` (e.g.
`python3.13`) and asks it, via a subprocess, to introspect the code
object with its own `dis`/`marshal` modules -- which are correct by
construction for that version. The result (`coderef.RemoteCode`)
plugs into the rest of the pipeline exactly like a real `CodeType`.
**No code from the analyzed program is ever executed**, only
introspected (`marshal.load` + `dis.get_instructions`).

If no matching interpreter is found, PyArch reports that plainly and
names the missing version, rather than guessing.

This has been verified against a real PyInstaller executable compiled
with Python 3.13.10, decompiled correctly while running under Python
3.12.

## Supported Python versions

Magic-number detection covers 3.8 through 3.14. Bytecode structuring
(control flow, functions, classes, imports, try/except) has been
tested against 3.12 and 3.13 bytecode. Versions where PyArch has no
matching interpreter available fall back to an honest error rather
than a silent, wrong result.

## Known limitations

- **List/set/dict comprehensions** (Python 3.12+'s inlined PEP 709
  shape) are reconstructed for a single `for` clause with at most one
  `if` filter. Nested comprehensions (`[x for row in m for x in row]`)
  and generator expressions are not yet matched and fall back to a
  diagnostic rather than invented code.
- **Function default values**: positional defaults (`def f(a, b=2)`)
  are recovered. Keyword-only defaults (`def f(a, *, b=5)`) and
  annotations are not yet handled.
- **`match`/`case`**: not implemented.
- **`async`/`await`/`async for`/`async with`**: not implemented.
- **Nested `try` inside `try`, or `try/except/else`**: falls back to
  a flat rendering of the try body with a diagnostic, rather than a
  (possibly wrong) nested structure.
- Pre-3.11 bytecode's `SETUP_FINALLY`-style exception handling (as
  opposed to 3.11+'s exception table) is not implemented; only the
  3.11+ mechanism is supported.
- No `tests/` suite yet (round-trip correctness for this session's
  work was verified by direct execution rather than a committed test
  suite).

## What's been verified

Round-trip (`compile()` -> decompile -> `ast.parse()` +
`compile()`) was checked by direct execution for: assignments,
arithmetic and comparisons, `if`/`elif`/`else` (including
CPython's loop-rotation and tail-return duplication shapes),
`while`, `for` (including `break`/`continue`), functions (including
nested functions, closures, `*args`/`**kwargs`, keyword-only
parameters, and positional default values), classes (including
inheritance and methods), attribute and subscript assignment, nested
function calls, `import`/`from ... import ... as ...`,
`try`/`except`/`finally` (bare, typed, multi-clause, with/without
`as`, inside a function with `return` in both the `try` and the
handler), and list/set/dict comprehensions (with and without an `if`
filter, at module and function scope). It was also verified
end-to-end against a real PyInstaller executable, compiled with
Python 3.13.10, via the CLI (`pyarch make`) while running under
Python 3.12.
