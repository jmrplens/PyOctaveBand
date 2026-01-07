#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Tests for backward compatibility and import fallbacks.
"""

import builtins
import importlib
import sys
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

import pytest


def test_import_literal_fallback() -> None:
    """
    Cover the 'except ImportError' branch for typing.Literal.
    This simulates an environment (like older Python) where Literal is missing
    from typing, forcing the code to try typing_extensions.
    """
    # 1. Unload modules to force re-import
    modules_to_unload = [
        "pyoctaveband", 
        "pyoctaveband.core", 
        "pyoctaveband.__init__"
    ]
    
    # Save original modules to restore later
    original_modules = {}
    for mod in modules_to_unload:
        if mod in sys.modules:
            original_modules[mod] = sys.modules.pop(mod)

    # 2. Mock import to fail for typing.Literal
    real_import = builtins.__import__

    def side_effect(
        name: str, 
        globals: Optional[Dict[str, Any]] = None, 
        locals: Optional[Dict[str, Any]] = None, 
        fromlist: Optional[Tuple[str, ...]] = (), 
        level: int = 0
    ) -> Any:
        if name == "typing" and fromlist is not None and "Literal" in fromlist:
            raise ImportError("Mocked ImportError for Literal")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=side_effect):
            # 3. Import should trigger the except block
            importlib.import_module("pyoctaveband.core")
            # Verify typing_extensions was used (we can't easily verify the variable source
            # without inspecting bytecode, but execution of the line is what coverage tracks)
            
            # Unload again to test __init__.py
            if "pyoctaveband" in sys.modules:
                del sys.modules["pyoctaveband"]
            importlib.import_module("pyoctaveband")

    except ImportError:
        pytest.fail("The fallback import failed (typing_extensions missing?)")
    finally:
        # Restore state
        for mod, value in original_modules.items():
            sys.modules[mod] = value
