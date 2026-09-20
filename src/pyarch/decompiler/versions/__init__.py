"""Version-specific bytecode profiles for Python 3.8–3.13."""

from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Any
import sys
import dis

@dataclass(frozen=True)
class BytecodeProfile:
    """Semantic profile for a specific Python version."""
    version: tuple  # (major, minor)
    opcode_map: Dict[int, str]  # opcode → semantic name
    has_extended_args: bool
    has_JUMP_BACKWARD: bool  # Python 3.11+
    has_PUSH_NULL: bool      # Python 3.11+
    has_LOAD_METHOD: bool    # Python 3.8–3.10
    has_LOAD_ATTR_METHOD: bool  # Python 3.12+
    has_CALL_INTRINSIC: bool  # Python 3.11+
    has_CALL_FUNCTION_EX: bool
    binary_ops: Dict[int, str]  # opcode → operator symbol
    compare_ops: Dict[int, str]
    # etc.

    def is_supported(self) -> bool:
        return 3 <= self.version[0] <= 3 and 8 <= self.version[1] <= 13

# We'll build profiles lazily from dis module when needed.
def get_profile(version: tuple) -> BytecodeProfile:
    """Return profile for given Python version (major, minor)."""
    # For now, we'll use the running Python's dis to infer, but we need to handle
    # different versions by parsing the magic number from .pyc.
    # This is a stub; we'll implement per-version mappings later.
    pass