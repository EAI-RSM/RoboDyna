import os as _os

# Task modules are organized into subfolders (household/, conceptual/) for
# readability, but the collector resolves every task by the flat name
# `envs.<task>` (importlib.import_module(f"envs.{task_name}")). Extend the
# package search path to include those subfolders so `envs.<task>` still
# resolves regardless of which subfolder the task file lives in — the task
# modules keep their `envs` package (relative imports like `from ._base_task`
# are unchanged) and no call site needs to know about the subfolders.
_here = _os.path.dirname(__file__)
for _sub in ("household", "conceptual"):
    _p = _os.path.join(_here, _sub)
    if _os.path.isdir(_p) and _p not in __path__:
        __path__.append(_p)

from .utils import *
from ._GLOBAL_CONFIGS import *
