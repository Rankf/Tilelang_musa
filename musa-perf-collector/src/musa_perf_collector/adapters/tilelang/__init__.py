"""TileLang-MUSA integration through public compiler APIs."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compiler_info import (
        TileLangCompilerInfoCollector,
        collect_compiler_information,
    )


def __getattr__(name: str) -> Any:
    """Load compiler collection lazily to avoid metadata-adapter cycles."""
    if name == "COMPILER_INFO_SCHEMA_VERSION":
        from .compiler_info import SCHEMA_VERSION

        return SCHEMA_VERSION
    if name in {"TileLangCompilerInfoCollector", "collect_compiler_information"}:
        from . import compiler_info

        return getattr(compiler_info, name)
    raise AttributeError(name)

__all__ = [
    "COMPILER_INFO_SCHEMA_VERSION",
    "TileLangCompilerInfoCollector",
    "collect_compiler_information",
]
