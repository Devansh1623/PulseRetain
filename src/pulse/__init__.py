"""
PulseRetain — SaaS Subscriber Attrition Intelligence Platform
src/pulse/__init__.py

Top-level package. Re-exports the config loader and version string.
"""

__version__ = "1.0.0"
__author__ = "PulseRetain Analytics Team"

from pulse.config import load_config  # noqa: F401
