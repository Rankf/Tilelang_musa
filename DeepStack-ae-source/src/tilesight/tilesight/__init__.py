"""Lightweight TileSight package entry point.

Optional NCU/plotting, Welder and fused-op helpers are intentionally not
imported here.  The historical eager imports made core architecture imports
require optional matplotlib/scipy packages.
"""

from .arch import *  # noqa: F401,F403

__all__ = []
