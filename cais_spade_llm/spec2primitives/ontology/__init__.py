"""Shared immutable ontology infrastructure for Spec2Primitives."""

from __future__ import annotations

from .ppr_tbox import (
    OntologyContextError,
    TBoxLoadError,
    TBoxProfileError,
    TBoxSnapshot,
    load_ppr_tbox,
)

__all__ = [
    "OntologyContextError",
    "TBoxLoadError",
    "TBoxProfileError",
    "TBoxSnapshot",
    "load_ppr_tbox",
]
