# Resource references

This directory is reserved for reusable, static resource references. The
authoritative primitive catalog remains resource-owned and is retrieved for the
exact RA instance through the implemented Spec2Primitives-owned Phase 5.1
adapter.

The adapter retrieves the complete versioned snapshot supplied by
that RA. No fixed primitive count is assumed, and every symbol remains exactly
as supplied. A repository reference may identify an approved catalog source,
but it never becomes a second catalog authority.

Phase 5.2A passes that exact snapshot with the reconstructed PA target feature
to the selected RobotAgent LLM for structural symbol selection. No catalog is
copied into this directory and no reference file becomes runtime authority.
