# PyArch

> **Statut : archivé / non maintenu activement, mais fonctionnel.**
> Ce projet n'est plus développé en continu. Il reste utilisable tel
> quel pour décompiler des programmes Python réels, avec les limites
> documentées ci-dessous. Les contributions/forks sont bienvenus.

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
structured reconstruction: if/while/for/try/with, functions, classes,
imports, comprehensions, f-strings, decorators
(decompiler/control_flow.py, statements.py, stack.py, functions.py,
classes.py, exception_table.py, comprehension_rewrite.py,
boolop_rewrite.py)
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

## Supported Python versions

Magic-number detection covers 3.8 through 3.14. Bytecode structuring
(control flow, functions, classes, imports, try/except/finally,
with-statements, comprehensions, f-strings, decorators) has been
tested against real 3.12 and 3.13 bytecode, including production code
(PyArch's own source). **Not every version/feature combination has
been exercised** -- see Known limitations. Versions where PyArch has
no matching interpreter available on `PATH` fail with an honest error
rather than a silent, wrong result.

## Known limitations

- **`match`/`case`**, **`async`/`await`/`async for`/`async with`**:
  not implemented.
- **`yield from`**: plain `yield` works; the iteration loop a
  `yield from` compiles to is not reconstructed.
- **Nested comprehensions** (`[x for row in m for x in row]`) and
  **generator expressions**: not matched; fall back to a diagnostic.
- **Keyword-only parameter defaults** (`def f(a, *, b=5)`): not
  reliably recovered (positional defaults work).
- **Decorator chains deeper than one level** (`@a\n@b\ndef f(): ...`)
  are only partially supported.
- **`CALL_FUNCTION_EX`** (calls built from `*args`/`**kwargs`
  unpacking) has not been specifically verified.
- **Nested `try` inside `try`, or `try/except/else`**: falls back to
  a flat rendering of the try body with a diagnostic.
- Pre-3.11 bytecode's `SETUP_FINALLY`-style exception handling (as
  opposed to 3.11+'s exception table) is not implemented.
- No `tests/` suite is committed; correctness has been checked by
  direct execution against real compiled bytecode throughout
  development, not by a standing regression suite.

## What's been verified

Round-trip (`compile()` -> decompile -> `ast.parse()` +
`compile()`) was checked by direct execution for: assignments,
arithmetic/comparisons (including Python 3.13's `bool(...)`-wrapped
comparisons), `if`/`elif`/`else` (including nested conditionals
inside loops and CPython's tail-test loop rotation), `while`, `for`
(including `break`/`continue`, tuple-unpacking targets), functions
(nested functions, closures, `*args`/`**kwargs`, keyword-only
parameters, positional defaults, `@property` and other single-level
decorators), classes (inheritance, `super()`, dataclass-style field
annotations), attribute/subscript assignment and slicing,
`raise`, nested function calls, `import`/`from ... import ... as
...`, `try`/`except`/`finally`, `with` statements, f-strings,
`and`/`or` as expressions (not just `if` conditions), list/set/dict
comprehensions, and basic generators (`yield`). It was also verified
end-to-end against a real PyInstaller executable compiled with
Python 3.13.10, decompiled while running under Python 3.12, and
against a piece of PyArch's own real source code.
