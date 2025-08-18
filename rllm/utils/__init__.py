"""Wrapper package to preserve legacy `rllm.utils` module (utils.py).

This directory exists only because we need a place for auxiliary helper
modules (e.g. kv_logger) without breaking existing code that expects
`import rllm.utils ...` to load the *module* implemented in
`rllm/utils.py`.

We load that legacy module dynamically and re-export its public
attributes so that both styles work:

    import rllm.utils              # gets legacy utils.py symbols
    from rllm.utils import kv_logger  # gets sub-modules placed here
"""
from __future__ import annotations

import importlib.util as _util
import importlib.machinery as _machinery
import os as _os
import sys as _sys
from types import ModuleType as _ModuleType
from pathlib import Path as _Path

# -----------------------------------------------------
# Load the original utils.py (now a sibling of *this* directory)
# -----------------------------------------------------
_pkg_dir = _Path(__file__).resolve().parent
_legacy_path = _pkg_dir.with_suffix(".py")  # rllm/utils.py

if _legacy_path.exists():
    _spec = _util.spec_from_loader(
        "rllm._legacy_utils",
        _machinery.SourceFileLoader("rllm._legacy_utils", str(_legacy_path)),
    )
    if _spec and _spec.loader:
        _legacy_mod = _util.module_from_spec(_spec)  # type: ignore[arg-type]
        _sys.modules[_spec.name] = _legacy_mod
        _spec.loader.exec_module(_legacy_mod)
        # Re-export public names (no leading underscore)
        globals().update({k: v for k, v in _legacy_mod.__dict__.items() if not k.startswith("_")})
else:
    # Fallback: nothing to export if the file went missing.
    _legacy_mod = None

# -----------------------------------------------------
# Clean up helper names
# -----------------------------------------------------
del _util, _machinery, _os, _sys, _Path, _ModuleType, _pkg_dir, _legacy_path, _spec