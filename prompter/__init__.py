"""
Prompter - A Python library for structured LLM prompting with multiple provider support
"""

from .schemas import *

from .base_executor import *

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
