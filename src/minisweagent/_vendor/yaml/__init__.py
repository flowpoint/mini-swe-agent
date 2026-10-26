"""PyYAML 6.0.3 safe loading only; see README.md and LICENSE."""

from .error import YAMLError
from .loader import SafeLoader

__all__ = ["YAMLError", "safe_load"]


def safe_load(stream):
    loader = SafeLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
