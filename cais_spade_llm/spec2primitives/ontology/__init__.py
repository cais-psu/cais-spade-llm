"""Shared immutable ontology infrastructure for Spec2Primitives."""

from __future__ import annotations

from .ppr_tbox import (
    OntologyContextError,
    TBoxLoadError,
    TBoxProfileError,
    TBoxSnapshot,
    load_ppr_tbox,
)
from .resource_registry import (
    ResourceRegistryEntry,
    ResourceRegistryError,
    ResourceRegistrySnapshot,
    load_predefined_resource_registry,
)
from .workcell import (
    PredefinedWorkcellError,
    PredefinedWorkcellSnapshot,
    load_predefined_workcell,
)

__all__ = [
    "OntologyContextError",
    "PredefinedWorkcellError",
    "PredefinedWorkcellSnapshot",
    "ResourceRegistryEntry",
    "ResourceRegistryError",
    "ResourceRegistrySnapshot",
    "TBoxLoadError",
    "TBoxProfileError",
    "TBoxSnapshot",
    "load_predefined_resource_registry",
    "load_predefined_workcell",
    "load_ppr_tbox",
]
