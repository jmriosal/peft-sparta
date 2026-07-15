import importlib

# map: class name -> submodule name
_LAZY = {
    "SFT":        "sft",
    "SFT_Config": "sft",
}

__all__ = list(_LAZY)

def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)

def __dir__():
    return sorted(__all__)


# Usage: from peft_sparta.trainer import SFT
#        imports ONLY sft.py (+ its dependencies)
