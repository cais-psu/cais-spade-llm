from __future__ import annotations

"""Define and validate the PA-owned Phase 4.3 grounding contracts.

The contracts in this module describe robot-independent product context.  They
do not select primitive operations, contact a resource, or treat an ontology
graph as an operational completeness oracle.
"""


import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rdflib import Literal, URIRef

from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationPresentationRecord,
    load_allocation_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellSnapshot,
    ResourceRegistrySnapshot,
    TBoxSnapshot,
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
    approved_document_metadata,
)

if TYPE_CHECKING:
    from .ontology_grounding import OntologyGroundingProposal

_BINDING_STATUSES = frozenset({"accepted", "ambiguous", "rejected", "stale", "unavailable"})
_SAFE_SYMBOL = re.compile(r"^[^\x00-\x1f\x7f]+$")
_VIEW_ROOT = Path("products/grounding/product_context")
_GROUNDING_PROPOSAL_KEYS = frozenset(
    {
        "record_type",
        "proposal_number",
        "initialized_specification_iri",
        "feature_iri",
        "output",
        "compiled_delta",
        "status",
        "failure",
        "grounding_evidence",
    }
)
_RECOGNITION_RECORD_TYPES = frozenset(
    {
        "DocumentSourceIndexRecord",
        "DocumentQueryRecord",
        "DocumentOverviewRecord",
        "CADGeometryRecord",
        "CADMeshRecord",
        "RGBDSegmentationRecord",
        "ObservationCandidateReview",
        "CADSizeCorrespondenceRecord",
        "CandidateSpatialRelationRecord",
        "RGBDObservationRecord",
        "RobotFrameLocationRecord",
        "CADPoseEstimationRecord",
    }
)
_RESOURCE_SELECTION_PREFIX = (
    "products",
    "grounding",
    "resource_selection",
)
_RESOURCE_SELECTION_KEYS = frozenset(
    {
        "motion_validation_performed",
        "process_symbol",
        "selected_resource_iri",
        "selected_execution_mode",
        "authority",
        "ontology_projection_ref",
        "evidence_presentation_fingerprint",
        "candidate_resource_iris",
        "validation_scope",
        "evidence_presentation_sha256",
        "ontology_projection_sha256",
        "allocation_presentation_ref",
        "fingerprint",
        "candidate_resource_symbols",
        "current_state_iri",
        "process_iri",
        "reachability_check_fingerprint",
        "reachability_check_sha256",
        "tbox_fingerprint",
        "specification_iri",
        "registry_fingerprint",
        "selection_number",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "workcell_fingerprint",
        "feature_iri",
        "reachability_check_ref",
        "record_type",
        "desired_state_iri",
        "selected_resource_symbol",
        "allocation_status",
        "state_locations",
        "evidence_presentation_ref",
        "selected_resource_jid",
    }
)
_DELTA_KEYS = frozenset(
    {
        "delta_number",
        "producer",
        "assertions",
        "uncertainty",
        "unresolved_evidence_needs",
        "typed_context_refs",
    }
)


class GroundingContractError(ValueError):
    """Raised when a Phase 4.3 grounding contract is invalid."""


class _EmbeddedEvidenceStateError(GroundingContractError):
    """Mark an unavailable or hash-changed embedded source as stale evidence."""


@dataclass(frozen=True)
class TypedContextBinding:
    """Summarize one validated non-RDF product-context record."""

    output_symbol: str
    subject_role: str
    record_type: str
    record_ref: str
    record_sha256: str
    status: str
    frame: str | None
    observed_at_ns: int | None
    valid_from_ns: int | None
    valid_until_ns: int | None
    producer: str
    evidence_refs: tuple[str, ...]
    provenance: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TypedContextBinding:
        """Validate and construct one persisted typed binding summary."""
        expected = {
            "output_symbol",
            "subject_role",
            "record_type",
            "record_ref",
            "record_sha256",
            "status",
            "frame",
            "observed_at_ns",
            "valid_from_ns",
            "valid_until_ns",
            "producer",
            "evidence_refs",
            "provenance",
        }
        _require_exact_keys(value, expected, "TypedContextBinding")
        status = _required_string(value["status"], "TypedContextBinding.status")
        if status not in _BINDING_STATUSES:
            raise GroundingContractError("TypedContextBinding.status is invalid.")
        provenance = _required_mapping(value["provenance"], "TypedContextBinding.provenance")
        return cls(
            output_symbol=_required_symbol(
                value["output_symbol"], "TypedContextBinding.output_symbol"
            ),
            subject_role=_required_symbol(
                value["subject_role"], "TypedContextBinding.subject_role"
            ),
            record_type=_required_symbol(value["record_type"], "TypedContextBinding.record_type"),
            record_ref=_required_string(value["record_ref"], "TypedContextBinding.record_ref"),
            record_sha256=_sha256_string(
                value["record_sha256"], "TypedContextBinding.record_sha256"
            ),
            status=status,
            frame=_optional_string(value["frame"], "TypedContextBinding.frame"),
            observed_at_ns=_optional_nonnegative_integer(
                value["observed_at_ns"], "TypedContextBinding.observed_at_ns"
            ),
            valid_from_ns=_optional_nonnegative_integer(
                value["valid_from_ns"], "TypedContextBinding.valid_from_ns"
            ),
            valid_until_ns=_optional_nonnegative_integer(
                value["valid_until_ns"], "TypedContextBinding.valid_until_ns"
            ),
            producer=_required_symbol(value["producer"], "TypedContextBinding.producer"),
            evidence_refs=_string_tuple(
                value["evidence_refs"], "TypedContextBinding.evidence_refs"
            ),
            provenance=dict(provenance),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe binding record."""
        return {
            "output_symbol": self.output_symbol,
            "subject_role": self.subject_role,
            "record_type": self.record_type,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "status": self.status,
            "frame": self.frame,
            "observed_at_ns": self.observed_at_ns,
            "valid_from_ns": self.valid_from_ns,
            "valid_until_ns": self.valid_until_ns,
            "producer": self.producer,
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ProductContextView:
    """Join the current PA ABox with validated typed-context bindings."""

    product_requirement: str
    interaction_namespace: str
    tbox_fingerprint: str
    abox_fingerprint: str
    delta_count: int
    assertions: tuple[Mapping[str, object], ...]
    typed_bindings: tuple[TypedContextBinding, ...]
    uncertainty: tuple[object, ...]
    unresolved_evidence_needs: tuple[object, ...]
    attempted_evidence: tuple[str, ...]
    assessed_at_ns: int
    fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProductContextView:
        """Validate and reconstruct one persisted view snapshot."""
        expected = {
            "record_type",
            "product_requirement",
            "interaction_namespace",
            "tbox_fingerprint",
            "abox_fingerprint",
            "delta_count",
            "assertions",
            "typed_bindings",
            "uncertainty",
            "unresolved_evidence_needs",
            "attempted_evidence",
            "assessed_at_ns",
            "fingerprint",
        }
        _require_exact_keys(value, expected, "ProductContextView")
        if value["record_type"] != "ProductContextView":
            raise GroundingContractError("ProductContextView identity is invalid.")
        assertions = value["assertions"]
        typed_bindings = value["typed_bindings"]
        uncertainty = value["uncertainty"]
        unresolved = value["unresolved_evidence_needs"]
        if not all(
            isinstance(item, list) for item in (assertions, typed_bindings, uncertainty, unresolved)
        ):
            raise GroundingContractError("ProductContextView collection fields are invalid.")
        result = cls(
            product_requirement=_required_string(
                value["product_requirement"], "ProductContextView.product_requirement"
            ),
            interaction_namespace=_required_string(
                value["interaction_namespace"], "ProductContextView.interaction_namespace"
            ),
            tbox_fingerprint=_sha256_string(
                value["tbox_fingerprint"], "ProductContextView.tbox_fingerprint"
            ),
            abox_fingerprint=_sha256_string(
                value["abox_fingerprint"], "ProductContextView.abox_fingerprint"
            ),
            delta_count=_nonnegative_integer(
                value["delta_count"], "ProductContextView.delta_count"
            ),
            assertions=tuple(
                dict(_required_mapping(item, "ProductContextView assertion")) for item in assertions
            ),
            typed_bindings=tuple(
                TypedContextBinding.from_mapping(
                    _required_mapping(item, "ProductContextView typed binding")
                )
                for item in typed_bindings
            ),
            uncertainty=tuple(uncertainty),
            unresolved_evidence_needs=tuple(unresolved),
            attempted_evidence=_string_tuple(
                value["attempted_evidence"], "ProductContextView.attempted_evidence"
            ),
            assessed_at_ns=_nonnegative_integer(
                value["assessed_at_ns"], "ProductContextView.assessed_at_ns"
            ),
            fingerprint=_sha256_string(value["fingerprint"], "ProductContextView.fingerprint"),
        )
        payload = result.to_record()
        payload.pop("fingerprint")
        if _fingerprint(payload) != result.fingerprint:
            raise GroundingContractError("ProductContextView fingerprint is invalid.")
        return result

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe view snapshot."""
        return {
            "record_type": "ProductContextView",
            "product_requirement": self.product_requirement,
            "interaction_namespace": self.interaction_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "abox_fingerprint": self.abox_fingerprint,
            "delta_count": self.delta_count,
            "assertions": [dict(item) for item in self.assertions],
            "typed_bindings": [item.to_record() for item in self.typed_bindings],
            "uncertainty": list(self.uncertainty),
            "unresolved_evidence_needs": list(self.unresolved_evidence_needs),
            "attempted_evidence": list(self.attempted_evidence),
            "assessed_at_ns": self.assessed_at_ns,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PAContextGroundingCompletion:
    """Hold one evidence-validated, MoveIt-validated Phase 4 completion."""

    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe validated completion bundle."""
        return dict(self.record)


@dataclass(frozen=True)
class _CompletionAllocationAuthority:
    """Hold candidate resources reconstructed from pinned registry/workcell records."""

    process_symbol: str
    process_iri: str
    candidate_resources: tuple[tuple[str, str, str], ...]
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str


@dataclass(frozen=True)
class GroundingProducerDescriptor:
    """Describe one provider's evidence capabilities without routing priority."""

    provider_id: str
    description: str
    accepted_evidence_types: tuple[str, ...]
    produced_record_types: tuple[str, ...]
    prerequisites: Mapping[str, tuple[str, ...]]
    availability: bool
    estimated_cost: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingProducerDescriptor:
        """Validate and construct one producer descriptor."""
        expected = {
            "provider_id",
            "description",
            "accepted_evidence_types",
            "produced_record_types",
            "prerequisites",
            "availability",
            "estimated_cost",
        }
        _require_exact_keys(value, expected, "GroundingProducerDescriptor")
        produced_record_types = _string_tuple(
            value["produced_record_types"], "produced_record_types"
        )
        if not produced_record_types:
            raise GroundingContractError(
                "GroundingProducerDescriptor.produced_record_types must be non-empty."
            )
        prerequisites_value = _required_mapping(value["prerequisites"], "prerequisites")
        if set(prerequisites_value) != set(produced_record_types):
            raise GroundingContractError(
                "GroundingProducerDescriptor.prerequisites must cover each output."
            )
        prerequisites = {
            record_type: _string_tuple(
                prerequisites_value[record_type],
                f"prerequisites[{record_type}]",
            )
            for record_type in produced_record_types
        }
        availability = value["availability"]
        if not isinstance(availability, bool):
            raise GroundingContractError(
                "GroundingProducerDescriptor.availability must be a boolean."
            )
        return cls(
            provider_id=_required_symbol(value["provider_id"], "provider_id"),
            description=_required_string(value["description"], "description"),
            accepted_evidence_types=_string_tuple(
                value["accepted_evidence_types"], "accepted_evidence_types"
            ),
            produced_record_types=produced_record_types,
            prerequisites=prerequisites,
            availability=availability,
            estimated_cost=_nonnegative_integer(value["estimated_cost"], "estimated_cost"),
        )

    @property
    def producer(self) -> str:
        """Return the provider ID used by existing controlled-tool boundaries."""
        return self.provider_id

    @property
    def evidence_types(self) -> tuple[str, ...]:
        """Return accepted evidence types for the existing serving boundary."""
        return self.accepted_evidence_types

    def supports_record_type(self, record_type: str) -> bool:
        """Return whether this provider can produce the requested typed record."""
        return record_type in self.produced_record_types

    def prerequisites_for(self, record_type: str) -> tuple[str, ...]:
        """Return prerequisites for one advertised output record."""
        return self.prerequisites.get(record_type, ())

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe descriptor record."""
        return {
            "provider_id": self.provider_id,
            "description": self.description,
            "accepted_evidence_types": list(self.accepted_evidence_types),
            "produced_record_types": list(self.produced_record_types),
            "prerequisites": {key: list(items) for key, items in self.prerequisites.items()},
            "availability": self.availability,
            "estimated_cost": self.estimated_cost,
        }


def build_product_context_view(
    interaction_root: Path,
    abox: ABoxSnapshot,
    *,
    attempted_evidence: Iterable[str],
    assessed_at_ns: int,
) -> ProductContextView:
    """Build a tamper-evident view from the current ABox and accepted deltas."""
    if not isinstance(assessed_at_ns, int) or isinstance(assessed_at_ns, bool):
        raise GroundingContractError("assessed_at_ns must be an integer.")
    root = Path(interaction_root).resolve()
    if abox.interaction_root.resolve() != root:
        raise GroundingContractError("ABox interaction root does not match the view root.")

    assertion_records = tuple(_abox_assertion_records(abox))
    abox_fingerprint = _fingerprint(assertion_records)
    bindings: list[TypedContextBinding] = []
    uncertainty: list[object] = []
    unresolved: list[object] = []
    seen_refs: set[str] = set()
    for delta_number in range(1, abox.delta_count + 1):
        delta = _read_delta(abox, delta_number)
        for caveat in delta["uncertainty"]:
            if caveat not in uncertainty:
                uncertainty.append(caveat)
        unresolved.extend(delta["unresolved_evidence_needs"])
        for record_ref in delta["typed_context_refs"]:
            if record_ref in seen_refs:
                continue
            seen_refs.add(record_ref)
            bindings.append(
                _typed_binding_from_ref(
                    root,
                    str(record_ref),
                    producer=str(delta["producer"]),
                )
            )

    attempted = tuple(
        _required_symbol(item, "attempted_evidence item") for item in attempted_evidence
    )
    payload = {
        "record_type": "ProductContextView",
        "product_requirement": abox.product_requirement,
        "interaction_namespace": abox.namespace,
        "tbox_fingerprint": abox.tbox_fingerprint,
        "abox_fingerprint": abox_fingerprint,
        "delta_count": abox.delta_count,
        "assertions": list(assertion_records),
        "typed_bindings": [item.to_record() for item in bindings],
        "uncertainty": uncertainty,
        "unresolved_evidence_needs": unresolved,
        "attempted_evidence": list(attempted),
        "assessed_at_ns": assessed_at_ns,
    }
    fingerprint = _fingerprint(payload)
    return ProductContextView(
        product_requirement=abox.product_requirement,
        interaction_namespace=abox.namespace,
        tbox_fingerprint=abox.tbox_fingerprint,
        abox_fingerprint=abox_fingerprint,
        delta_count=abox.delta_count,
        assertions=assertion_records,
        typed_bindings=tuple(bindings),
        uncertainty=tuple(uncertainty),
        unresolved_evidence_needs=tuple(unresolved),
        attempted_evidence=attempted,
        assessed_at_ns=assessed_at_ns,
        fingerprint=fingerprint,
    )


def persist_product_context_view(
    interaction_root: Path,
    view: ProductContextView,
) -> Path:
    """Persist one append-only ProductContextView snapshot."""
    root = Path(interaction_root).resolve()
    view_root = root / _VIEW_ROOT
    numbers = [
        int(path.stem.removeprefix("view_"))
        for path in view_root.glob("view_*.json")
        if path.stem.removeprefix("view_").isdigit()
    ]
    view_number = max(numbers, default=-1) + 1
    path = view_root / f"view_{view_number:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(view.to_record(), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise GroundingContractError(f"ProductContextView already exists: {path.name}.") from exc
    return path


def _finite_vector(value: object, length: int, *, positive: bool = False) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and (not positive or float(item) > 0.0)
            for item in value
        )
    )


def _accepted_typed_binding_matches(
    product_context: ProductContextView,
    *,
    record_type: str,
    record_ref: str,
    record_sha256: str,
) -> bool:
    return (
        sum(
            binding.record_type == record_type
            and binding.status == "accepted"
            and binding.record_ref == record_ref
            and binding.record_sha256 == record_sha256
            for binding in product_context.typed_bindings
        )
        == 1
    )


def _validated_fingerprinted_record(
    value: Mapping[str, object], *, expected_type: str, label: str
) -> str:
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if (
        (value.get("record_type") != expected_type)
        or (not _is_sha256_value(fingerprint))
        or (_fingerprint(payload) != fingerprint)
    ):
        raise GroundingContractError(f"{label} fingerprint is invalid.")
    return str(fingerprint)


def _completion_allocation_authority(
    registry: Mapping[str, object],
    workcell: Mapping[str, object],
    *,
    process_iri: str,
) -> _CompletionAllocationAuthority:
    """Reconstruct capable candidates from hash-pinned registry/workcell snapshots."""
    registry_expected = {
        "record_type",
        "ppr_namespace",
        "resource_namespace",
        "tbox_fingerprint",
        "workcell_profile_sha256",
        "graph_fingerprint",
        "resources",
        "fingerprint",
    }
    workcell_expected = {
        "record_type",
        "processes",
        "resource_iris",
        "resource_capabilities",
        "ppr_namespace",
        "resource_namespace",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_profile_sha256",
        "graph_fingerprint",
        "fingerprint",
    }
    _require_exact_keys(registry, registry_expected, "ResourceRegistrySnapshot ")
    _require_exact_keys(workcell, workcell_expected, "PredefinedWorkcellSnapshot ")
    registry_fingerprint = _validated_fingerprinted_record(
        registry,
        expected_type="ResourceRegistrySnapshot",
        label="ResourceRegistrySnapshot ",
    )
    workcell_fingerprint = _validated_fingerprinted_record(
        workcell,
        expected_type="PredefinedWorkcellSnapshot",
        label="PredefinedWorkcellSnapshot ",
    )
    tbox_fingerprint = _sha256_string(
        registry.get("tbox_fingerprint"),
        "registry tbox_fingerprint",
    )
    if (
        workcell.get("tbox_fingerprint") != tbox_fingerprint
        or workcell.get("registry_fingerprint") != registry_fingerprint
        or workcell.get("ppr_namespace") != registry.get("ppr_namespace")
        or workcell.get("resource_namespace") != registry.get("resource_namespace")
        or workcell.get("workcell_profile_sha256") != registry.get("workcell_profile_sha256")
    ):
        raise GroundingContractError("Pinned registry/workcell lineage is inconsistent.")

    raw_resources = registry.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise GroundingContractError("Pinned registry resources are invalid.")
    resources: list[tuple[str, str, str]] = []
    for item in raw_resources:
        resource = _required_mapping(item, "registry resource")
        _require_exact_keys(
            resource,
            {
                "resource_symbol",
                "resource_iri",
                "resource_jid",
                "resource_type",
                "source_ref",
                "source_sha256",
            },
            "registry resource",
        )
        if resource.get("resource_type") != "robot" or not _is_sha256_value(
            resource.get("source_sha256")
        ):
            raise GroundingContractError("Pinned registry resource is invalid.")
        resources.append(
            (
                _required_string(resource["resource_symbol"], "resource_symbol"),
                _required_string(resource["resource_iri"], "resource_iri"),
                _required_string(resource["resource_jid"], "resource_jid"),
            )
        )
    if len({item[0] for item in resources}) != len(resources) or len(
        {item[1] for item in resources}
    ) != len(resources):
        raise GroundingContractError("Pinned registry resource identities are not unique.")
    if workcell.get("resource_iris") != [item[1] for item in resources]:
        raise GroundingContractError("Pinned workcell resources do not match the registry.")

    raw_processes = workcell.get("processes")
    if not isinstance(raw_processes, list) or not raw_processes:
        raise GroundingContractError("Pinned workcell processes are invalid.")
    processes: list[tuple[str, str]] = []
    for item in raw_processes:
        process = _required_mapping(item, "workcell process")
        _require_exact_keys(
            process,
            {"process_symbol", "process_iri"},
            "workcell process",
        )
        processes.append(
            (
                _required_string(process["process_symbol"], "process_symbol"),
                _required_string(process["process_iri"], "process_iri"),
            )
        )
    process_matches = [item for item in processes if item[1] == process_iri]
    if len(process_matches) != 1:
        raise GroundingContractError("Selected process is not uniquely pinned by the workcell.")

    raw_capabilities = workcell.get("resource_capabilities")
    if not isinstance(raw_capabilities, list):
        raise GroundingContractError("Pinned workcell capabilities are invalid.")
    capabilities: dict[str, tuple[str, ...]] = {}
    for item in raw_capabilities:
        capability = _required_mapping(item, "resource capability")
        _require_exact_keys(
            capability,
            {"resource_iri", "capable_process_iris"},
            "resource capability",
        )
        resource_iri = _required_string(capability["resource_iri"], "resource_iri")
        process_iris = _string_tuple(
            capability["capable_process_iris"],
            "capable_process_iris",
        )
        if resource_iri in capabilities:
            raise GroundingContractError("Pinned workcell capability is duplicated.")
        capabilities[resource_iri] = process_iris
    if set(capabilities) != {item[1] for item in resources}:
        raise GroundingContractError("Pinned workcell capability coverage is invalid.")
    candidates = tuple(
        resource for resource in resources if process_iri in capabilities[resource[1]]
    )
    if not candidates:
        raise GroundingContractError("Pinned workcell has no capable resource for the process.")
    return _CompletionAllocationAuthority(
        process_symbol=process_matches[0][0],
        process_iri=process_iri,
        candidate_resources=candidates,
        tbox_fingerprint=tbox_fingerprint,
        registry_fingerprint=registry_fingerprint,
        workcell_fingerprint=workcell_fingerprint,
    )


def _reviewed_resolved_values(
    root: Path | None,
    target: Mapping[str, object],
    view: ProductContextView,
) -> list[Mapping[str, object]]:
    from .ontology_grounding import resolve_json_pointer

    resolved = []
    bindings = {item.record_ref: item for item in view.typed_bindings if item.status == "accepted"}
    for state in ("current_state", "desired_state"):
        names = set()
        for value in target[state]["state_values"]:
            ref = value["value_ref"]
            binding = bindings.get(ref["record_ref"])
            if binding is None or value["name"] in names:
                raise GroundingContractError(
                    "Bound state value is not an accepted unique binding."
                )
            names.add(value["name"])
            if root is not None:
                path = _completion_ref_path(root, ref["record_ref"], prefix=("products",))
                if _sha256_path(path) != binding.record_sha256:
                    raise GroundingContractError("Bound state value evidence changed.")
                resolve_json_pointer(
                    _read_json_mapping(path, "bound state value"), ref["field_path"]
                )
            resolved.append(
                {
                    "state": state,
                    "name": value["name"],
                    "value_ref": ref,
                    "record_type": binding.record_type,
                    "record_sha256": binding.record_sha256,
                }
            )
    return resolved


def source_uncertainty(
    target_feature: Mapping[str, object],
    records: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Retain exact upstream caveats from cited records and bound observation candidates."""
    from .ontology_grounding import target_feature_evidence_refs

    cited = set(target_feature_evidence_refs(target_feature))
    selected: dict[str, set[str]] = {}
    for state in ("current_state", "desired_state"):
        for value in target_feature[state]["state_values"]:
            ref = value["value_ref"]
            selected.setdefault(ref["record_ref"], set()).add(ref["field_path"])
    converted_candidates: dict[tuple[str, str, str], list[str]] = {}
    for record_ref in selected:
        record = records.get(record_ref, {})
        source = record.get("source_segmentation", record.get("segmentation"))
        if isinstance(source, Mapping):
            source = source.get("record", source)
        candidate = record.get("candidate_reference")
        if isinstance(source, Mapping) and isinstance(candidate, Mapping):
            identity = (
                source.get("ref"),
                candidate.get("observation_handle"),
                candidate.get("candidate_handle"),
            )
            if all(isinstance(item, str) for item in identity):
                converted_candidates.setdefault(identity, []).append(record_ref)
    result: list[dict[str, object]] = []

    def append(description: object, refs: Sequence[str]) -> None:
        if isinstance(description, str) and description.strip():
            entry = {"description": description, "evidence_refs": list(refs)}
            if entry not in result:
                result.append(entry)

    def collect(value: object, ref: str) -> None:
        if isinstance(value, Mapping):
            caveats = value.get("uncertainty", [])
            if isinstance(caveats, str):
                append(caveats, [ref])
            elif isinstance(caveats, list):
                for caveat in caveats:
                    if isinstance(caveat, Mapping):
                        append(caveat.get("description"), caveat.get("evidence_refs", [ref]))
                    else:
                        append(caveat, value.get("evidence_refs", [ref]))
            for key, nested in value.items():
                if key != "uncertainty":
                    collect(nested, ref)
        elif isinstance(value, list):
            for nested in value:
                collect(nested, ref)

    for ref, record in records.items():
        if record.get("record_type") == "ObservationCandidateReview":
            segmentation_ref = record["source_segmentation"]["ref"]
            for candidate in record["candidates"]:
                identity = (
                    segmentation_ref,
                    candidate.get("observation_handle"),
                    candidate.get("candidate_handle"),
                )
                converted_refs = converted_candidates.get(identity, [])
                if (
                    candidate["source_field_path"] in selected.get(segmentation_ref, set())
                    or converted_refs
                ):
                    append(candidate.get("uncertainty"), [ref, segmentation_ref, *converted_refs])
        elif ref in cited or cited.intersection(record.get("evidence_refs", [])):
            collect(record, ref)
    return result


def _grounding_artifact_path(root: Path, ref: str) -> Path:
    path = (root / ref).resolve()
    if not path.is_relative_to(root) and not path.is_relative_to(root.parent / "source_cache"):
        raise GroundingContractError("Grounding artifact leaves the authorized source roots.")
    return path


def _pin_grounding_artifact(
    root: Path, artifacts: dict[str, str], ref: str, expected: str | None = None
) -> Path:
    path = _grounding_artifact_path(root, ref)
    actual = _sha256_path(path)
    if (expected is not None and actual != expected) or (
        ref in artifacts and artifacts[ref] != actual
    ):
        raise GroundingContractError("Grounding source artifact changed.")
    artifacts[ref] = actual
    return path


def _pin_grounding_dependencies(root: Path, value: object, artifacts: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        ref, sha = value.get("ref"), value.get("sha256")
        if isinstance(ref, str) and isinstance(sha, str):
            if ref in artifacts and artifacts[ref] != sha:
                raise GroundingContractError("Grounding source dependency hash is inconsistent.")
            if ref not in artifacts:
                path = _pin_grounding_artifact(root, artifacts, ref, sha)
                if path.suffix == ".json":
                    _pin_grounding_dependencies(root, json.loads(path.read_text()), artifacts)
        for nested in value.values():
            _pin_grounding_dependencies(root, nested, artifacts)
    elif isinstance(value, list):
        for nested in value:
            _pin_grounding_dependencies(root, nested, artifacts)


def _pin_grounding_document_pages(root: Path, path: Path, artifacts: dict[str, str]) -> None:
    from ...tools.document_evidence.interpreter import (
        _load_interaction_source_index,
        _rendered_pages_from_source_index,
    )

    _, _, snapshot = _load_interaction_source_index(root, path)
    _pin_grounding_artifact(root, artifacts, snapshot["cache_record_ref"])
    for page in _rendered_pages_from_source_index(snapshot):
        _pin_grounding_artifact(root, artifacts, str(page.image_path), page.image_sha256)


def _validate_grounding_native_metadata(root: Path, value: object) -> None:
    if isinstance(value, Mapping):
        ref = value.get("context_ref", value.get("document_context_ref"))
        expected = value.get("source_sha256")
        if (
            isinstance(ref, str)
            and isinstance(expected, str)
            and _native_source_hash(root, ref, product_requirement="") != expected
        ):
            raise GroundingContractError("Grounding approved native source changed.")
        for nested in value.values():
            _validate_grounding_native_metadata(root, nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_grounding_native_metadata(root, nested)


def _grounding_request(root: Path, proposal_number: int) -> Mapping[str, object]:
    requests = sorted(
        (root / "products/grounding/ontology_grounding").glob(
            f"request_{proposal_number:04d}_*.json"
        )
    )
    if not requests:
        raise GroundingContractError("Grounding proposal requires its original PA request.")
    return _read_json_mapping(requests[-1], "grounding request")


def _grounding_clarification_refs(
    root: Path, request: Mapping[str, object], requirement: str
) -> tuple[set[str], Path | None]:
    from .presentation_records import load_evidence_presentation

    try:
        prompt = _required_string(request.get("prompt"), "grounding request prompt")
        payload = json.loads(prompt.rsplit("Grounding input:\n", 1)[1])
        entries = [
            item
            for item in payload["approved_evidence_catalog"]
            if _required_mapping(item, "grounding catalog item").get("evidence_type")
            == "user_clarification"
        ]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise GroundingContractError("Original PA grounding input is invalid.") from exc
    refs: set[str] = set()
    presentation = None
    for entry in entries:
        supplied_ref = entry.get("evidence_ref")
        if not isinstance(supplied_ref, str):
            raise GroundingContractError("Supplied clarification reference is invalid.")
        if supplied_ref.startswith("citation_") and presentation is None:
            presentation = load_evidence_presentation(root)
        matches = []
        for path in (root / "interaction_record").glob("clarification_*.json"):
            ref = path.relative_to(root).as_posix()
            expected = (
                presentation.opaque_reference(ref, kind="citation")
                if supplied_ref.startswith("citation_")
                else ref
            )
            if supplied_ref == expected:
                matches.append((path, ref))
        if len(matches) != 1:
            raise GroundingContractError("Supplied clarification source is unavailable.")
        path, ref = matches[0]
        record = _read_json_mapping(path, "supplied clarification")
        number = _nonnegative_integer(record.get("question_turn"), "clarification question_turn")
        if (
            record.get("record_type") != "PAClarification"
            or record.get("action") != "answered"
            or record.get("product_requirement") != requirement
            or not isinstance(entry.get("reply"), str)
            or not entry["reply"]
            or record.get("reply") != entry["reply"]
            or number == 0
            or path.name != f"clarification_{number:04d}.json"
        ):
            raise GroundingContractError(
                "Supplied clarification does not match its original reply."
            )
        refs.add(ref)
    return refs, None if presentation is None else presentation.record_path


def _grounding_issued_refs(
    root: Path, view: ProductContextView, proposal_number: int
) -> tuple[set[str], Path | None]:
    # Discovery catalog metadata grants no citation authority. Accepted retrieval
    # roots and explicitly supplied user replies are independent of the proposal.
    refs = {"requirement_0001"}
    for binding in view.typed_bindings:
        if binding.status == "accepted":
            refs.add(binding.record_ref)
            refs.update(binding.evidence_refs)
    clarification_refs, presentation_path = _grounding_clarification_refs(
        root, _grounding_request(root, proposal_number), view.product_requirement
    )
    refs.update(clarification_refs)
    return refs, presentation_path


def build_grounding_evidence(
    root: Path,
    *,
    context_view_ref: str,
    target_feature: Mapping[str, object],
    proposal_number: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Pin one immutable PA evidence snapshot and retain its source-owned caveats.

    The snapshot supplies the complete original typed-record root set. Later
    allocation views cannot change that set or silently drop an uncited caveat.
    """
    from ...tools.observation_presentation import PRESENTATION_REF, ObservationPresentation
    from .ontology_grounding import target_feature_evidence_refs

    root = Path(root).resolve()
    context_path = _completion_ref_path(root, context_view_ref, prefix=_VIEW_ROOT.parts)
    view = ProductContextView.from_mapping(_read_json_mapping(context_path, "grounding context"))
    artifacts: dict[str, str] = {}
    _pin_grounding_artifact(root, artifacts, context_view_ref)
    requirement_path = _pin_grounding_artifact(
        root, artifacts, "products/user_requirement/product_requirement.json"
    )
    if _read_json_mapping(requirement_path, "requirement").get("product_requirement") != (
        view.product_requirement
    ):
        raise GroundingContractError("Grounding requirement does not match its source record.")
    records: dict[str, Mapping[str, object]] = {}
    seen_refs: set[str] = set()
    for binding in view.typed_bindings:
        if binding.record_ref in seen_refs:
            raise GroundingContractError("Grounding context contains duplicate typed records.")
        seen_refs.add(binding.record_ref)
        current = _typed_binding_from_ref(root, binding.record_ref, producer=binding.producer)
        if current != binding:
            raise GroundingContractError(
                "Grounding typed binding changed after its context snapshot."
            )
        path = _pin_grounding_artifact(root, artifacts, binding.record_ref, binding.record_sha256)
        record = _read_json_mapping(path, "grounding typed record")
        if record.get("record_type") != binding.record_type:
            raise GroundingContractError("Grounding typed-record identity changed.")
        if binding.record_type not in _RECOGNITION_RECORD_TYPES:
            continue
        _validate_grounding_native_metadata(root, record)
        _pin_grounding_dependencies(root, record, artifacts)
        records[binding.record_ref] = record
        if binding.record_type == "DocumentSourceIndexRecord":
            _pin_grounding_document_pages(root, path, artifacts)
    if (root / PRESENTATION_REF).is_file():
        ObservationPresentation(root).assert_unchanged()
        _pin_grounding_artifact(root, artifacts, PRESENTATION_REF)
    for pattern in (
        f"products/grounding/ontology_grounding/request_{proposal_number:04d}_*.json",
        "interaction_record/model_tool_exchange_*.json",
    ):
        for path in sorted(root.glob(pattern)):
            _pin_grounding_artifact(root, artifacts, path.relative_to(root).as_posix())
    refs = tuple(dict.fromkeys(target_feature_evidence_refs(target_feature)))
    issued_refs, presentation_path = _grounding_issued_refs(root, view, proposal_number)
    if not set(refs).issubset(issued_refs):
        raise GroundingContractError("Grounding proposal cites evidence that was not issued.")
    if presentation_path is not None:
        _pin_grounding_artifact(root, artifacts, presentation_path.relative_to(root).as_posix())
    manifest = {
        "source_refs": [
            {
                "ref": ref,
                "sha256": _native_source_hash(
                    root, ref, product_requirement=view.product_requirement
                ),
            }
            for ref in refs
        ],
        "input_artifacts": [{"ref": ref, "sha256": sha} for ref, sha in sorted(artifacts.items())],
    }
    return manifest, source_uncertainty(target_feature, records)


def _grounding_snapshot_ref(root: Path, manifest: Mapping[str, object]) -> str:
    _require_exact_keys(manifest, {"source_refs", "input_artifacts"}, "grounding_evidence")
    snapshots = []
    for field in ("source_refs", "input_artifacts"):
        entries = manifest[field]
        if not isinstance(entries, list):
            raise GroundingContractError("Grounding evidence entries are invalid.")
        refs: set[str] = set()
        for item in entries:
            if not isinstance(item, Mapping) or set(item) != {"ref", "sha256"}:
                raise GroundingContractError("Grounding evidence entry is invalid.")
            ref = _required_string(item["ref"], "grounding evidence ref")
            _sha256_string(item["sha256"], "grounding evidence sha256")
            if ref in refs:
                raise GroundingContractError("Grounding evidence contains duplicate references.")
            refs.add(ref)
            if field == "input_artifacts":
                if _sha256_path(_grounding_artifact_path(root, ref)) != item["sha256"]:
                    raise GroundingContractError("Grounding evidence artifact changed.")
                if Path(ref).parts[:3] == _VIEW_ROOT.parts:
                    snapshots.append(ref)
    if len(snapshots) != 1:
        raise GroundingContractError("Grounding evidence requires one original context snapshot.")
    return snapshots[0]


def _validated_grounding_target(
    root: Path, projection: Mapping[str, object], snapshot: ProductContextView
) -> OntologyGroundingProposal:
    from .ontology_grounding import (
        OntologyGroundingProposal,
        _validated_assembly_feature_association,
        _validated_evidence_refs,
        _validated_state,
    )

    target = projection["output"]["target_feature"]
    process = _required_mapping(target.get("required_process"), "required_process")
    _require_exact_keys(process, {"process_iri", "evidence_refs"}, "required_process")
    request = _grounding_request(root, projection["proposal_number"])
    branches = request["response_format"]["schema"]["properties"]["result"]["anyOf"]
    schemas = [
        branch["properties"]["target_feature"]
        for branch in branches
        if "target_feature" in branch.get("properties", {})
    ]
    matching = [
        schema
        for schema in schemas
        if process.get("process_iri")
        in schema["properties"]["required_process"]["properties"]["process_iri"]["enum"]
    ]
    if len(matching) != 1 or set(target) != set(matching[0]["required"]):
        raise GroundingContractError("Grounding target fields or authorized process are invalid.")
    authorized, _ = _grounding_issued_refs(root, snapshot, projection["proposal_number"])
    _validated_evidence_refs(process["evidence_refs"], "required_process.evidence_refs", authorized)
    bindings = {
        item.record_ref: item for item in snapshot.typed_bindings if item.status == "accepted"
    }

    def resolve_record(ref: str) -> Mapping[str, object]:
        binding = bindings.get(ref)
        if binding is None:
            raise GroundingContractError("Grounding state value is not an accepted typed binding.")
        return {
            "record_type": binding.record_type,
            "record_sha256": binding.record_sha256,
            "record": _read_json_mapping(_grounding_artifact_path(root, ref), "state evidence"),
        }

    resolved = []
    for state in ("current_state", "desired_state"):
        resolved.extend(
            _validated_state(
                state,
                target[state],
                authorized_evidence_refs=authorized,
                typed_record_resolver=resolve_record,
            )
        )
    associations = target.get("assembly_feature_association")
    if associations is not None:
        if not isinstance(associations, list):
            raise GroundingContractError("Grounding associations must be a collection.")
        for association in associations:
            _validated_assembly_feature_association(
                association,
                resolved_state_values=resolved,
                authorized_evidence_refs=authorized,
            )
    return OntologyGroundingProposal(
        target_feature=target,
        feature_iri=f"{snapshot.interaction_namespace}feature_0001",
        resolved_state_values=tuple(resolved),
        evidence_refs=tuple(authorized),
        assembly_feature_association=associations,
    )


def _grounding_ppr_namespace(snapshot: ProductContextView) -> str:
    specification = f"{snapshot.interaction_namespace}specification_1"
    types = [
        item["object"]["value"]
        for item in snapshot.assertions
        if item.get("subject") == specification
        and item.get("predicate") == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        and item.get("object", {}).get("kind") == "iri"
    ]
    if len(types) != 1 or not types[0].endswith("specification"):
        raise GroundingContractError("Grounding snapshot specification type is invalid.")
    return types[0].removesuffix("specification")


def validate_grounding_evidence(
    root: Path,
    projection: Mapping[str, object],
    *,
    require_committed: bool = True,
) -> ProductContextView:
    """Recheck exact PA proposal provenance without judging the meaning of its claims."""
    from .ontology_grounding import _compile_proposal_delta

    root = Path(root).resolve()
    _require_exact_keys(projection, set(_GROUNDING_PROPOSAL_KEYS), "OntologyGroundingProposal")
    if (
        projection["record_type"] != "OntologyGroundingProposal"
        or projection["status"] != "accepted"
        or projection["failure"] is not None
    ):
        raise GroundingContractError("Grounding evidence requires one accepted PA proposal.")
    proposal_number = _nonnegative_integer(projection["proposal_number"], "proposal_number")
    if proposal_number == 0:
        raise GroundingContractError("Grounding proposal number must be positive.")
    output = _required_mapping(projection["output"], "grounding output")
    _require_exact_keys(output, {"target_feature"}, "grounding output")
    target = _required_mapping(output["target_feature"], "target_feature")
    manifest = _required_mapping(projection["grounding_evidence"], "grounding_evidence")
    snapshot_ref = _grounding_snapshot_ref(root, manifest)
    snapshot = ProductContextView.from_mapping(
        _read_json_mapping(root / snapshot_ref, "grounding context")
    )
    try:
        proposal = _validated_grounding_target(root, projection, snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroundingContractError("Grounding target structure or binding is invalid.") from exc
    expected, caveats = build_grounding_evidence(
        root, context_view_ref=snapshot_ref, target_feature=target, proposal_number=proposal_number
    )
    if manifest != expected:
        raise GroundingContractError("Grounding evidence coverage or source hashes changed.")
    if (
        projection["initialized_specification_iri"]
        != f"{snapshot.interaction_namespace}specification_1"
    ):
        raise GroundingContractError("Grounding proposal specification changed.")
    if projection["feature_iri"] != proposal.feature_iri:
        raise GroundingContractError("Grounding proposal feature identity changed.")
    delta = _required_mapping(projection["compiled_delta"], "grounding compiled delta")
    compiled = _compile_proposal_delta(
        proposal,
        namespace=snapshot.interaction_namespace,
        specification_iri=projection["initialized_specification_iri"],
        ppr_namespace=_grounding_ppr_namespace(snapshot),
    )
    if delta != {**compiled, "uncertainty": caveats}:
        raise GroundingContractError("Grounding compiled assertions or source uncertainty changed.")
    if require_committed:
        _validate_grounding_delta_linkage(root, projection, snapshot)
    return snapshot


def _validate_grounding_delta_linkage(
    root: Path, projection: Mapping[str, object], snapshot: ProductContextView
) -> None:
    number = snapshot.delta_count + 1
    path = root / "products/grounding/ontology" / f"delta_{number:04d}.json"
    delta = _read_json_mapping(path, "accepted grounding delta")
    expected = {
        "delta_number": number,
        "producer": "ontology_grounding",
        **projection["compiled_delta"],
    }
    if delta != expected:
        raise GroundingContractError("Grounding snapshot does not precede its accepted delta.")
    typed_refs: list[str] = []
    for previous in range(1, number):
        path = root / "products/grounding/ontology" / f"delta_{previous:04d}.json"
        record = _read_json_mapping(path, "grounding evidence delta")
        _require_exact_keys(record, set(_DELTA_KEYS), "grounding evidence delta")
        if record["delta_number"] != previous:
            raise GroundingContractError("Grounding evidence delta order changed.")
        for ref in record["typed_context_refs"]:
            if ref not in typed_refs:
                typed_refs.append(ref)
    if typed_refs != [binding.record_ref for binding in snapshot.typed_bindings]:
        raise GroundingContractError("Grounding snapshot dropped original typed evidence.")


def _validate_reviewed_projection(
    root: Path,
    projection: Mapping[str, object],
    view: ProductContextView,
) -> None:
    try:
        snapshot = validate_grounding_evidence(root, projection)
        if snapshot.product_requirement != view.product_requirement:
            raise ValueError("Grounding requirement changed.")
        if any(item not in view.uncertainty for item in projection["compiled_delta"]["uncertainty"]):
            raise ValueError("Source uncertainty is missing from accepted context.")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise GroundingContractError("Grounding proposal or source lineage is invalid.") from exc


def _live_location_constraints() -> dict[str, list[str]]:
    return {
        "checked_constraints": [
            "process_capability",
            "live_robot_state",
            "joint_limits",
            "collision_aware_position_plans",
            "all_reviewed_state_locations",
        ],
        "unvalidated_constraints": [
            "grasp",
            "orientation",
            "attached_part_collision_geometry",
            "insertion",
            "force",
            "tolerance",
            "primitive_composition",
        ],
    }


def _validate_reviewed_location_selection(
    root: Path,
    projection: Mapping[str, object],
    view: ProductContextView,
    selection: Mapping[str, object],
    reachability: Mapping[str, object],
) -> None:
    """Revalidate the current selection against every bound location and its provenance."""
    expected_keys = _RESOURCE_SELECTION_KEYS
    _require_exact_keys(selection, expected_keys, "ResourceSelectionRecord")
    _validated_fingerprinted_record(
        selection,
        expected_type="ResourceSelectionRecord",
        label="ResourceSelectionRecord",
    )
    _validated_fingerprinted_record(
        reachability,
        expected_type="ReachabilityCheckRecord",
        label="ReachabilityCheckRecord",
    )
    if (
        (projection.get("status") != "accepted")
        or (selection.get("authority") != "ProductAgent")
        or (selection.get("allocation_status") != "accepted")
        or (selection.get("validation_scope") != ("moveit_state_location_reachability"))
        or (selection.get("motion_validation_performed") is not True)
        or (reachability.get("status") != "accepted")
        or (reachability.get("authority") != "ProductAgent.check_reachability")
    ):
        raise GroundingContractError(
            "Current assignment requires grounded goal and reachability evidence."
        )
    projection_path = _completion_ref_path(
        root,
        selection["ontology_projection_ref"],
        prefix=("products", "grounding", "ontology_grounding"),
    )
    if (
        _sha256_path(projection_path) != selection["ontology_projection_sha256"]
        or _read_json_mapping(projection_path, "proposal") != projection
    ):
        raise GroundingContractError("Selection proposal linkage changed.")
    presentation = load_allocation_presentation(root)
    presentation.assert_unchanged()
    for prefix, path in (
        ("allocation_presentation", presentation.record_path),
        ("evidence_presentation", root / presentation.evidence_presentation_ref),
    ):
        record = _read_json_mapping(path, prefix)
        if (
            selection[f"{prefix}_sha256"] != _sha256_path(path)
            or selection[f"{prefix}_fingerprint"] != record["fingerprint"]
            or selection[f"{prefix}_ref"] != path.relative_to(root).as_posix()
        ):
            raise GroundingContractError("Selection presentation lineage changed.")
    for field in (
        "feature_iri",
        "process_iri",
        "process_symbol",
        "specification_iri",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
    ):
        if selection[field] != reachability[field]:
            raise GroundingContractError("Selection reachability authority changed.")
    for field in ("resource_symbol", "resource_iri", "resource_jid", "execution_mode"):
        if selection[f"selected_{field}"] != reachability[field]:
            raise GroundingContractError("Selected arm differs from the checked arm.")
    selected_identity = tuple(
        selection[f"selected_{field}"]
        for field in ("resource_symbol", "resource_iri", "resource_jid")
    )
    if selected_identity not in {
        (entry.resource_symbol, entry.resource_iri, entry.resource_jid)
        for entry in presentation.resources
    }:
        raise GroundingContractError("Selected arm was not presented.")
    _validate_reviewed_location_check(
        root,
        projection,
        view,
        presentation,
        selection["state_locations"],
        {state: selection[f"{state}_iri"] for state in ("current_state", "desired_state")},
        reachability,
    )


def _validate_reviewed_location_check(
    root: Path,
    projection: Mapping[str, object],
    view: ProductContextView,
    presentation: AllocationPresentationRecord,
    state_locations: Mapping[str, object],
    state_iris: Mapping[str, str],
    reachability: Mapping[str, object],
) -> None:
    """Validate any arm's accepted, rejected, or unavailable result against bound positions."""
    _require_exact_keys(
        reachability,
        {
            "record_type",
            "check_number",
            "authority",
            "specification_iri",
            "feature_iri",
            "process_symbol",
            "process_iri",
            "resource_symbol",
            "resource_iri",
            "resource_jid",
            "execution_mode",
            "target_frame",
            "manifest_ref",
            "manifest_sha256",
            "allocation_presentation_ref",
            "allocation_presentation_sha256",
            "allocation_presentation_fingerprint",
            "state_locations",
            "tbox_fingerprint",
            "registry_fingerprint",
            "workcell_fingerprint",
            "status",
            "validation",
            "fingerprint",
        },
        "ReachabilityCheckRecord",
    )
    from .resource_grounding import (
        _load_robot_frame_location,
        _resource_environment,
        _location_planning_profile,
        _state_location_evidence,
        validate_location_planning_request,
        validate_location_planning_response,
    )

    from ...config import load_workcell_profile

    profile = load_workcell_profile()
    resources = [
        entry
        for entry in profile.resources
        if entry.iri == reachability["resource_iri"]
        and entry.symbol == reachability["resource_symbol"]
        and entry.manifest_ref == reachability["manifest_ref"]
        and reachability["process_iri"] in entry.capable_process_iris
    ]
    if len(resources) != 1:
        raise GroundingContractError("Selected resource capability is not authorized.")
    manifest_path = resources[0].manifest_path
    if _sha256_path(manifest_path) != reachability["manifest_sha256"]:
        raise GroundingContractError("Reachability capability manifest changed.")
    manifest = _read_json_mapping(manifest_path, "resource manifest")
    environment = _resource_environment(
        manifest[reachability["resource_symbol"]], reachability["resource_symbol"]
    )
    validation = reachability.get("validation")
    if (
        not isinstance(validation, Mapping)
        or set(validation) != {"request", "response", "validated_at_ns"}
        or type(validation["validated_at_ns"]) is not int
        or validation["validated_at_ns"] <= 0
    ):
        raise GroundingContractError("MoveIt validation record is invalid.")
    request = validation["request"]
    validate_location_planning_request(request)
    validate_location_planning_response(request, validation["response"])
    if validation["response"]["status"] != reachability["status"]:
        raise GroundingContractError("MoveIt result status differs from the reachability record.")
    for key in (
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "target_frame",
    ):
        if request[key] != reachability[key]:
            raise GroundingContractError("MoveIt planning authority changed.")
    for key, expected in _location_planning_profile(
        environment, reachability["target_frame"]
    ).items():
        if request[key] != expected:
            raise GroundingContractError("MoveIt controller profile changed.")
    target = projection["output"]["target_feature"]
    values = _reviewed_resolved_values(root, target, view)
    groups = reachability.get("state_locations")
    if not isinstance(groups, Mapping) or set(groups) != {"current_state", "desired_state"}:
        raise GroundingContractError("Reachability must cover both product states.")
    for state_name in ("current_state", "desired_state"):
        expected = {
            (
                value["value_ref"]["record_ref"],
                value["record_sha256"],
                value["value_ref"]["field_path"],
            )
            for value in values
            if value["state"] == state_name
            and value["record_type"] in {"RGBDSegmentationRecord", "RobotFrameLocationRecord"}
        }
        state_values = groups[state_name]
        actual = {
            (value["source_record_ref"], value["source_record_sha256"], value["source_field_path"])
            for value in state_values
        }
        if (
            not expected
            or expected != actual
            or len(actual) != len(state_values)
            or state_locations[state_name] != [value["evidence_handle"] for value in state_values]
        ):
            raise GroundingContractError(
                "Reachability omitted or substituted a bound state location."
            )
        for value, result in zip(
            state_values, validation["response"]["state_locations"][state_name], strict=True
        ):
            entry = presentation.evidence_for_handle(value["evidence_handle"])
            assignment = {
                "state_iri": state_iris[state_name],
                "evidence_handle": entry.pa_handle,
                "source_record_type": entry.record_type,
                "source_record_ref": entry.record_ref,
                "source_record_sha256": entry.record_sha256,
                "source_field_path": entry.field_path,
            }
            if any(
                value.get(key) != expected for key, expected in assignment.items()
            ) or not _accepted_typed_binding_matches(
                view,
                record_type="RobotFrameLocationRecord",
                record_ref=value["location_record_ref"],
                record_sha256=value["location_record_sha256"],
            ):
                raise GroundingContractError(
                    "MoveIt location binding differs from accepted evidence."
                )
            location = _load_robot_frame_location(
                root / value["location_record_ref"], root, target_frame=reachability["target_frame"]
            )
            calculated = _state_location_evidence(
                state_name,
                state_iris[state_name],
                location,
                evidence=entry,
                reachable=result["status"] == "accepted",
            )
            if calculated.to_record() != value:
                raise GroundingContractError(
                    "Reachability result differs from the pinned geometry and capability evidence."
                )

    expected_locations = {
        state: [
            {
                key: item[key]
                for key in (
                    "state_iri",
                    "evidence_handle",
                    "translation_m",
                    "location_record_ref",
                    "location_record_sha256",
                )
            }
            for item in items
        ]
        for state, items in groups.items()
    }
    if request["state_locations"] != expected_locations:
        raise GroundingContractError("MoveIt planned for different positions or evidence.")


def _validate_resource_check_coverage(
    root: Path,
    projection: Mapping[str, object],
    view: ProductContextView,
    selection: Mapping[str, object],
    tool_refs: Sequence[Mapping[str, object]],
    authority: _CompletionAllocationAuthority,
) -> None:
    """Recheck every capable arm using only pinned calls from this allocation attempt."""
    presentation = load_allocation_presentation(root)
    presentation.assert_unchanged()
    candidates = {item[0]: item for item in authority.candidate_resources}
    if set(presentation.resource_order) != set(candidates):
        raise GroundingContractError("Allocation presentation omitted a capable resource.")
    checked: set[str] = set()
    selected_ref_seen = False
    for pinned in tool_refs:
        call_path = _completion_ref_path(root, pinned["ref"], prefix=("interaction_record",))
        if _sha256_path(call_path) != pinned["sha256"]:
            raise GroundingContractError("Allocation tool-call evidence changed.")
        call = _read_json_mapping(call_path, "allocation tool call")
        if call.get("record_type") != "ProductAgentAllocationToolCall":
            continue
        _require_exact_keys(
            call,
            {
                "record_type",
                "tool_call_id",
                "tool_name",
                "arguments",
                "result_ref",
                "result_sha256",
                "result",
                "failure",
            },
            "ProductAgentAllocationToolCall",
        )
        if call["tool_name"] != "check_reachability":
            raise GroundingContractError("Allocation used an unauthorized tool.")
        arguments = _required_mapping(call["arguments"], "allocation arguments")
        _require_exact_keys(arguments, {"resource_symbol"}, "allocation arguments")
        symbol = arguments["resource_symbol"]
        if symbol not in candidates:
            raise GroundingContractError("Allocation checked an unauthorized resource.")
        if call["result_ref"] is None:
            # An invalid call has no validated planning outcome and cannot satisfy coverage.
            continue
        if symbol in checked or call["failure"] is not None:
            raise GroundingContractError("Allocation has conflicting resource checks.")
        check_path = _completion_ref_path(
            root,
            call["result_ref"],
            prefix=("products", "grounding", "reachability"),
        )
        if _sha256_path(check_path) != call["result_sha256"]:
            raise GroundingContractError("Allocation reachability evidence changed.")
        check = _read_json_mapping(check_path, "ReachabilityCheckRecord")
        _validated_fingerprinted_record(
            check, expected_type="ReachabilityCheckRecord", label="ReachabilityCheckRecord"
        )
        if (
            check.get("authority") != "ProductAgent.check_reachability"
            or (check.get("resource_symbol"), check.get("resource_iri"), check.get("resource_jid"))
            != candidates[symbol]
        ):
            raise GroundingContractError("Allocation resource-check authority changed.")
        for field in (
            "feature_iri",
            "process_iri",
            "process_symbol",
            "specification_iri",
            "tbox_fingerprint",
            "registry_fingerprint",
            "workcell_fingerprint",
            "allocation_presentation_ref",
            "allocation_presentation_sha256",
            "allocation_presentation_fingerprint",
        ):
            if check.get(field) != selection[field]:
                raise GroundingContractError(
                    "Allocation resource checks use different validated context."
                )
        result = _required_mapping(call["result"], "allocation tool result")
        from .resource_proximity import validate_allocation_proximity

        validate_allocation_proximity(result, check)
        if (
            result.get("status") != check["status"]
            or result.get("resource_symbol") != symbol
            or result.get("reachability_check_ref")
            != f"reachability_check_{check['check_number']:04d}"
        ):
            raise GroundingContractError("Allocation tool result differs from its pinned check.")
        _validate_reviewed_location_check(
            root,
            projection,
            view,
            presentation,
            selection["state_locations"],
            {state: selection[f"{state}_iri"] for state in ("current_state", "desired_state")},
            check,
        )
        checked.add(symbol)
        selected_ref_seen |= call["result_ref"] == selection["reachability_check_ref"]
    if checked != set(candidates) or not selected_ref_seen:
        missing = sorted(set(candidates) - checked)
        raise GroundingContractError(
            f"Allocation must check every capable resource. Unchecked resources: {', '.join(missing)}. Start a fresh interaction."
        )


def persist_pa_context_grounding_completion(
    interaction_root: Path,
    *,
    tbox: TBoxSnapshot,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    ontology_projection_ref: str,
    resource_selection_ref: str,
    tool_call_refs: Sequence[str],
    registry: ResourceRegistrySnapshot | None = None,
    workcell: PredefinedWorkcellSnapshot | None = None,
) -> Path:
    """Persist one current validated completion lineage."""
    proposal_label = "OntologyGroundingProposal"
    root = Path(interaction_root).resolve()
    if product_context.product_requirement != product_requirement:
        raise GroundingContractError(
            "Completion requirement does not match the final ProductContextView."
        )
    tbox.assert_unchanged()
    selected_registry = registry
    if selected_registry is None and workcell is not None:
        selected_registry = workcell._registry
    if selected_registry is None:
        selected_registry = load_predefined_resource_registry(tbox)
    selected_workcell = workcell or load_predefined_workcell(tbox, selected_registry)
    selected_registry.assert_unchanged()
    selected_workcell.assert_unchanged()

    projection_path = _completion_ref_path(
        root,
        ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    projection = _read_json_mapping(projection_path, proposal_label)
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    if (
        (
            set(projection)
            != (
                {
                    "record_type",
                    "proposal_number",
                    "initialized_specification_iri",
                    "feature_iri",
                    "output",
                    "compiled_delta",
                    "status",
                    "failure",
                }
                | {"grounding_evidence"}
            )
        )
        or (projection.get("record_type") != "OntologyGroundingProposal")
        or (projection.get("status") != "accepted")
        or (projection.get("failure") is not None)
    ):
        raise GroundingContractError(f"{proposal_label} is invalid.")
    output = _required_mapping(projection.get("output"), "proposal output")
    _require_exact_keys(output, {"target_feature"}, "proposal output")
    target_feature = _required_mapping(output["target_feature"], "target_feature")
    process = _required_mapping(target_feature["required_process"], "required_process")
    process_iri = _required_string(process.get("process_iri"), "process_iri")
    process_symbol = selected_workcell.process_symbol_for_iri(process_iri)
    target_feature_keys = {"required_process", "current_state", "desired_state"}
    if process_symbol == "assembly":
        target_feature_keys.add("assembly_feature_association")
    _require_exact_keys(target_feature, target_feature_keys, "target_feature")
    feature_iri = _required_string(projection.get("feature_iri"), "feature_iri")
    specification_iri = _required_string(
        projection.get("initialized_specification_iri"),
        "initialized_specification_iri",
    )
    compiled_delta = _required_mapping(projection.get("compiled_delta"), "compiled_delta")
    compiled_assertions = compiled_delta.get("assertions")
    if not isinstance(compiled_assertions, list):
        raise GroundingContractError(f"{proposal_label} has an invalid compiled assertion count.")

    _validate_reviewed_projection(root, projection, product_context)
    selection_payload = dict(selection)
    selection_fingerprint = selection_payload.pop("fingerprint", None)
    selection_expected = {
        "record_type",
        "selection_number",
        "authority",
        "specification_iri",
        "feature_iri",
        "process_symbol",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
        "candidate_resource_iris",
        "candidate_resource_symbols",
        "state_locations",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "allocation_status",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "fingerprint",
    }
    selection_expected.update(
        {
            "ontology_projection_ref",
            "ontology_projection_sha256",
            "validation_scope",
            "motion_validation_performed",
        }
    )
    state_locations = selection.get("state_locations")
    if (
        (set(selection) != selection_expected)
        or (selection.get("record_type") != "ResourceSelectionRecord")
        or (selection.get("authority") != "ProductAgent")
        or (selection.get("allocation_status") != "accepted")
        or (selection.get("specification_iri") != specification_iri)
        or (selection.get("feature_iri") != feature_iri)
        or (selection.get("process_iri") != process_iri)
        or (not isinstance(state_locations, Mapping))
        or (set(state_locations) != {"current_state", "desired_state"})
        or (
            any(
                not isinstance(state_locations[state_name], list)
                or not state_locations[state_name]
                or not all(
                    isinstance(handle, str) and handle for handle in state_locations[state_name]
                )
                for state_name in ("current_state", "desired_state")
            )
        )
        or (not _is_sha256_value(selection_fingerprint))
        or (_fingerprint(selection_payload) != selection_fingerprint)
    ):
        raise GroundingContractError("ResourceSelectionRecord is invalid.")

    registry_record = selected_registry.to_record()
    workcell_record = selected_workcell.to_record()
    allocation_authority = _completion_allocation_authority(
        registry_record,
        workcell_record,
        process_iri=process_iri,
    )
    selected_identity = (
        selection.get("selected_resource_symbol"),
        selection.get("selected_resource_iri"),
        selection.get("selected_resource_jid"),
    )
    if (
        selected_identity not in allocation_authority.candidate_resources
        or selection.get("process_symbol") != allocation_authority.process_symbol
        or selection.get("tbox_fingerprint") != tbox.fingerprint
        or selection.get("registry_fingerprint") != selected_registry.fingerprint
        or selection.get("workcell_fingerprint") != selected_workcell.fingerprint
    ):
        raise GroundingContractError(
            "ResourceSelectionRecord  does not match the pinned authorities."
        )

    reachability_ref = _required_string(
        selection.get("reachability_check_ref"),
        "reachability_check_ref",
    )
    reachability_path = _completion_ref_path(
        root,
        reachability_ref,
        prefix=("products", "grounding", "reachability"),
    )
    reachability = _read_json_mapping(reachability_path, "ReachabilityCheckRecord")
    reach_payload = dict(reachability)
    reach_fingerprint = reach_payload.pop("fingerprint", None)
    if (
        (reachability.get("record_type") != "ReachabilityCheckRecord")
        or (reachability.get("status") != "accepted")
        or (reachability.get("resource_symbol") != selection.get("selected_resource_symbol"))
        or (reachability.get("resource_iri") != selection.get("selected_resource_iri"))
        or (reachability.get("resource_jid") != selection.get("selected_resource_jid"))
        or (reachability.get("process_iri") != process_iri)
        or (reachability.get("feature_iri") != feature_iri)
        or (not _is_sha256_value(reach_fingerprint))
        or (_fingerprint(reach_payload) != reach_fingerprint)
        or (selection.get("reachability_check_sha256") != _sha256_path(reachability_path))
        or (selection.get("reachability_check_fingerprint") != reach_fingerprint)
    ):
        raise GroundingContractError("ReachabilityCheckRecord is invalid.")
    reach_groups = reachability.get("state_locations")
    reach_handles = (
        {
            state_name: [
                item.get("evidence_handle") if isinstance(item, Mapping) else None
                for item in reach_groups.get(state_name, [])
            ]
            for state_name in ("current_state", "desired_state")
        }
        if isinstance(reach_groups, Mapping)
        else None
    )
    if reach_handles != state_locations:
        raise GroundingContractError("ReachabilityCheckRecord is invalid.")

    validation_ref = ontology_projection_ref
    validation_scope = "moveit_state_location_reachability"
    allocation_label = "resource assignment validated by MoveIt"
    validation = _live_location_constraints()
    _validate_reviewed_location_selection(
        root, projection, product_context, selection, reachability
    )

    assignment_delta_ref = _native_assignment_delta_ref(
        root,
        resource_selection_ref=resource_selection_ref,
        reachability_ref=reachability_ref,
        validation_ref=validation_ref,
        selected_resource_iri=_required_string(
            selection.get("selected_resource_iri"),
            "selected_resource_iri",
        ),
    )
    assignment_delta_path = _completion_ref_path(
        root,
        assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    typed_ref_names = {str(item["ref"]) for item in typed_refs}
    cited_refs: list[str] = []

    def collect_evidence_refs(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "evidence_refs" and isinstance(item, list):
                    cited_refs.extend(ref for ref in item if isinstance(ref, str) and ref)
                else:
                    collect_evidence_refs(item)
        elif isinstance(value, list):
            for item in value:
                collect_evidence_refs(item)

    collect_evidence_refs(target_feature)
    source_names = tuple(dict.fromkeys(ref for ref in cited_refs if ref not in typed_ref_names))
    source_refs = tuple(
        {
            "ref": source_ref,
            "sha256": _native_source_hash(
                root,
                source_ref,
                product_requirement=product_requirement,
            ),
        }
        for source_ref in source_names
    )
    tool_refs = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
        }
        for ref in dict.fromkeys(tool_call_refs)
    )
    _validate_resource_check_coverage(
        root,
        projection,
        product_context,
        selection,
        tool_refs,
        allocation_authority,
    )

    completion_root = root / "products/grounding/completion"
    registry_path = completion_root / "resource_registry_snapshot_0001.json"
    workcell_path = completion_root / "predefined_workcell_snapshot_0001.json"
    _write_json_mapping_exclusive(registry_path, registry_record)
    _write_json_mapping_exclusive(workcell_path, workcell_record)
    evidence_presentation_ref = _required_string(
        selection.get("evidence_presentation_ref"),
        "evidence_presentation_ref",
    )
    evidence_presentation_path = _completion_ref_path(
        root,
        evidence_presentation_ref,
        prefix=("products", "grounding", "presentation"),
    )
    allocation_presentation_ref = _required_string(
        selection.get("allocation_presentation_ref"),
        "allocation_presentation_ref",
    )
    allocation_presentation_path = _completion_ref_path(
        root,
        allocation_presentation_ref,
        prefix=("products", "grounding", "presentation"),
    )
    decision_path = _completion_ref_path(root, decision_ref, prefix=("interaction_record",))
    final_abox_path = root / "products/grounding/ontology/interaction_abox.ttl"
    if not final_abox_path.is_file():
        raise GroundingContractError("Final interaction ABox is unavailable.")
    payload: dict[str, object] = {
        "record_type": "PAContextGroundingCompletion",
        "status": "grounding complete",
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        "decision_sha256": _sha256_path(decision_path),
        "process_symbol": allocation_authority.process_symbol,
        "process_iri": process_iri,
        "feature_iri": feature_iri,
        "current_state_iri": selection["current_state_iri"],
        "desired_state_iri": selection["desired_state_iri"],
        "ontology_projection_ref": ontology_projection_ref,
        "ontology_projection_sha256": _sha256_path(projection_path),
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_selection_fingerprint": selection_fingerprint,
        "reachability_check_ref": reachability_ref,
        "reachability_check_sha256": _sha256_path(reachability_path),
        "reachability_check_fingerprint": reach_fingerprint,
        "resource_assignment_delta_ref": assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(assignment_delta_path),
        "final_abox_ref": final_abox_path.relative_to(root).as_posix(),
        "final_abox_sha256": _sha256_path(final_abox_path),
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "registry_snapshot_ref": registry_path.relative_to(root).as_posix(),
        "registry_snapshot_sha256": _sha256_path(registry_path),
        "registry_snapshot_fingerprint": selected_registry.fingerprint,
        "workcell_snapshot_ref": workcell_path.relative_to(root).as_posix(),
        "workcell_snapshot_sha256": _sha256_path(workcell_path),
        "workcell_snapshot_fingerprint": selected_workcell.fingerprint,
        "evidence_presentation_ref": evidence_presentation_ref,
        "evidence_presentation_sha256": _sha256_path(evidence_presentation_path),
        "evidence_presentation_fingerprint": selection["evidence_presentation_fingerprint"],
        "allocation_presentation_ref": allocation_presentation_ref,
        "allocation_presentation_sha256": _sha256_path(allocation_presentation_path),
        "allocation_presentation_fingerprint": selection["allocation_presentation_fingerprint"],
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
        "validation_scope": validation_scope,
        "checked_constraints": validation["checked_constraints"],
        "unvalidated_constraints": validation["unvalidated_constraints"],
        "allocation_label": allocation_label,
        "motion_executed": False,
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["motion_validation_performed"] = True
    payload["fingerprint"] = _fingerprint(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, payload)
    load_pa_context_grounding_completion(root)
    return completion_path


def load_pa_context_grounding_completion(
    interaction_root: Path,
) -> PAContextGroundingCompletion:
    """Load the single current completion shape and verify its entire evidence chain."""
    root = Path(interaction_root).resolve()
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise GroundingContractError("Exactly one PAContextGroundingCompletion record is required.")
    value = _read_json_mapping(paths[0], "PAContextGroundingCompletion")
    return PAContextGroundingCompletion(_validated_two_decision_completion(root, value))


def load_completed_product_context_view(interaction_root: Path) -> ProductContextView:
    """Reload the final ProductContextView pinned by the validated completion."""
    root = Path(interaction_root).resolve()
    completion = load_pa_context_grounding_completion(root)
    view = _latest_product_context_view(root)
    if view.abox_fingerprint != completion.to_record()["abox_fingerprint"]:
        raise GroundingContractError("Completed ProductContextView changed.")
    return view


def _validated_two_decision_completion(
    root: Path, value: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate the current validated completion and its entire pinned lineage."""
    completion_label = "PAContextGroundingCompletion"
    proposal_label = "OntologyGroundingProposal"
    expected = {
        "record_type",
        "status",
        "product_requirement",
        "completion_turn",
        "decision_ref",
        "decision_sha256",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "current_state_iri",
        "desired_state_iri",
        "ontology_projection_ref",
        "ontology_projection_sha256",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "final_abox_ref",
        "final_abox_sha256",
        "tbox_fingerprint",
        "abox_fingerprint",
        "registry_snapshot_ref",
        "registry_snapshot_sha256",
        "registry_snapshot_fingerprint",
        "workcell_snapshot_ref",
        "workcell_snapshot_sha256",
        "workcell_snapshot_fingerprint",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "typed_context_refs",
        "source_refs",
        "tool_call_refs",
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
        "allocation_label",
        "motion_executed",
        "completed_at_ns",
        "fingerprint",
    }
    expected.add("motion_validation_performed")
    _require_exact_keys(value, expected, completion_label)
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    validation_scope = value.get("validation_scope")
    expected_allocation_label = "resource assignment validated by MoveIt"
    if (
        (value.get("record_type") != "PAContextGroundingCompletion")
        or (value.get("status") != "grounding complete")
        or validation_scope != "moveit_state_location_reachability"
        or (value.get("allocation_label") != expected_allocation_label)
        or (value.get("motion_executed") is not False)
        or (not _is_sha256_value(fingerprint))
        or (_fingerprint(payload) != fingerprint)
    ):
        raise GroundingContractError(f"{completion_label} is invalid.")

    pinned = (
        ("decision_ref", "decision_sha256", ("interaction_record",)),
        (
            "ontology_projection_ref",
            "ontology_projection_sha256",
            ("products", "grounding", "ontology_grounding"),
        ),
        (
            "resource_selection_ref",
            "resource_selection_sha256",
            _RESOURCE_SELECTION_PREFIX,
        ),
        (
            "reachability_check_ref",
            "reachability_check_sha256",
            ("products", "grounding", "reachability"),
        ),
        (
            "resource_assignment_delta_ref",
            "resource_assignment_delta_sha256",
            ("products", "grounding", "ontology"),
        ),
        (
            "final_abox_ref",
            "final_abox_sha256",
            ("products", "grounding", "ontology"),
        ),
        (
            "registry_snapshot_ref",
            "registry_snapshot_sha256",
            ("products", "grounding", "completion"),
        ),
        (
            "workcell_snapshot_ref",
            "workcell_snapshot_sha256",
            ("products", "grounding", "completion"),
        ),
        (
            "evidence_presentation_ref",
            "evidence_presentation_sha256",
            ("products", "grounding", "presentation"),
        ),
        (
            "allocation_presentation_ref",
            "allocation_presentation_sha256",
            ("products", "grounding", "presentation"),
        ),
    )
    resolved: dict[str, Path] = {}
    for ref_field, sha_field, prefix in pinned:
        ref = _required_string(value.get(ref_field), ref_field)
        path = _completion_ref_path(root, ref, prefix=prefix)
        if value.get(sha_field) != _sha256_path(path):
            raise GroundingContractError(f"{completion_label} pinned {ref_field} changed.")
        resolved[ref_field] = path

    for field in ("typed_context_refs", "tool_call_refs"):
        entries = value.get(field)
        if not isinstance(entries, list):
            raise GroundingContractError(f"{completion_label} {field} is invalid.")
        for item in entries:
            if not isinstance(item, Mapping) or set(item) != {"ref", "sha256"}:
                raise GroundingContractError(f"{completion_label} {field} entry is invalid.")
            ref = _required_string(item.get("ref"), f"{field}.ref")
            path = _completion_ref_path(
                root,
                ref,
                prefix=("interaction_record",) if field == "tool_call_refs" else ("products",),
            )
            if item.get("sha256") != _sha256_path(path):
                raise GroundingContractError(f"{completion_label} {field} changed.")
    source_refs = value.get("source_refs")
    if not isinstance(source_refs, list):
        raise GroundingContractError(f"{completion_label} source_refs are invalid.")
    product_requirement = _required_string(
        value.get("product_requirement"),
        "product_requirement",
    )
    for item in source_refs:
        if not isinstance(item, Mapping) or set(item) != {"ref", "sha256"}:
            raise GroundingContractError(f"{completion_label} source ref is invalid.")
        source_ref = _required_string(item.get("ref"), "source_ref")
        if item.get("sha256") != _native_source_hash(
            root,
            source_ref,
            product_requirement=product_requirement,
        ):
            raise GroundingContractError(f"{completion_label} source evidence changed.")

    projection = _read_json_mapping(
        resolved["ontology_projection_ref"],
        proposal_label,
    )
    selection = _read_json_mapping(
        resolved["resource_selection_ref"],
        "ResourceSelectionRecord",
    )
    _validate_requested_resource(resolved["decision_ref"], selection)
    reachability = _read_json_mapping(
        resolved["reachability_check_ref"],
        "ReachabilityCheckRecord",
    )
    if (
        (projection.get("status") != "accepted")
        or (selection.get("allocation_status") != "accepted")
        or (reachability.get("status") != "accepted")
        or (selection.get("process_iri") != value.get("process_iri"))
        or (selection.get("feature_iri") != value.get("feature_iri"))
        or (reachability.get("resource_iri") != selection.get("selected_resource_iri"))
    ):
        raise GroundingContractError(f"{completion_label} decision lineage is inconsistent.")
    final_view = _latest_product_context_view(root)
    if (
        final_view.product_requirement != product_requirement
        or final_view.tbox_fingerprint != value.get("tbox_fingerprint")
        or final_view.abox_fingerprint != value.get("abox_fingerprint")
    ):
        raise GroundingContractError(f"{completion_label} final context is inconsistent.")
    _validate_reviewed_projection(root, projection, final_view)
    _validate_reviewed_location_selection(root, projection, final_view, selection, reachability)
    authority = _completion_allocation_authority(
        _read_json_mapping(resolved["registry_snapshot_ref"], "registry"),
        _read_json_mapping(resolved["workcell_snapshot_ref"], "workcell"),
        process_iri=value["process_iri"],
    )
    _validate_resource_check_coverage(
        root, projection, final_view, selection, value["tool_call_refs"], authority
    )
    if (
        authority.process_symbol != value["process_symbol"]
        or authority.tbox_fingerprint != value["tbox_fingerprint"]
        or authority.registry_fingerprint != value["registry_snapshot_fingerprint"]
        or authority.workcell_fingerprint != value["workcell_snapshot_fingerprint"]
        or selection["registry_fingerprint"] != authority.registry_fingerprint
        or selection["workcell_fingerprint"] != authority.workcell_fingerprint
        or (
            selection["selected_resource_symbol"],
            selection["selected_resource_iri"],
            selection["selected_resource_jid"],
        )
        not in authority.candidate_resources
        or set(selection["candidate_resource_iris"])
        != {item[1] for item in authority.candidate_resources}
        or set(selection["candidate_resource_symbols"])
        != {item[0] for item in authority.candidate_resources}
    ):
        raise GroundingContractError("Current assignment capability authority changed.")
    for field in (
        "feature_iri",
        "process_iri",
        "process_symbol",
        "current_state_iri",
        "desired_state_iri",
        "tbox_fingerprint",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
    ):
        if value[field] != selection[field]:
            raise GroundingContractError("Completion assignment linkage changed.")
    if (
        value.get("motion_validation_performed") is not True
        or validation_scope != "moveit_state_location_reachability"
        or any(value.get(key) != expected for key, expected in _live_location_constraints().items())
    ):
        raise GroundingContractError("MoveIt allocation validation scope changed.")
    if (
        value.get("resource_selection_fingerprint") != selection["fingerprint"]
        or value.get("reachability_check_fingerprint") != reachability["fingerprint"]
    ):
        raise GroundingContractError("Assignment fingerprint linkage changed.")
    assignment_ref = _native_assignment_delta_ref(
        root,
        resource_selection_ref=value["resource_selection_ref"],
        reachability_ref=value["reachability_check_ref"],
        validation_ref=value["ontology_projection_ref"],
        selected_resource_iri=selection["selected_resource_iri"],
    )
    if assignment_ref != value["resource_assignment_delta_ref"]:
        raise GroundingContractError("Assignment ontology lineage changed.")
    return dict(value)


def _is_sha256_value(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_requested_resource(decision_path: Path, selection: Mapping[str, object]) -> None:
    """Keep a rebuilt completion bound to its explicit user resource constraint."""
    decision = _read_json_mapping(decision_path, "allocation decision")
    decision_input = decision.get("PA_input")
    if (
        isinstance(decision_input, Mapping)
        and decision_input.get("requested_resource_symbol") is not None
        and decision_input["requested_resource_symbol"] != selection["selected_resource_symbol"]
    ):
        raise GroundingContractError("Selected resource differs from the explicit user request.")


def _latest_product_context_view(root: Path) -> ProductContextView:
    paths = sorted((root / _VIEW_ROOT).glob("view_*.json"))
    if not paths:
        raise GroundingContractError(
            "PAContextGroundingCompletion requires a final ProductContextView."
        )
    return ProductContextView.from_mapping(
        _read_json_mapping(paths[-1], "final ProductContextView")
    )


def _iri_local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _read_delta(abox: ABoxSnapshot, delta_number: int) -> Mapping[str, Any]:
    path = abox.ontology_root / f"delta_{delta_number:04d}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(
            f"Accepted ontology delta could not be read: {path.name}."
        ) from exc
    if (
        (not isinstance(value, dict))
        or (set(value) != _DELTA_KEYS)
        or (value["delta_number"] != delta_number)
        or (not isinstance(value["producer"], str))
        or (not isinstance(value["uncertainty"], list))
        or (not isinstance(value["unresolved_evidence_needs"], list))
        or (not isinstance(value["typed_context_refs"], list))
        or (not all(isinstance(item, str) for item in value["typed_context_refs"]))
    ):
        raise GroundingContractError(f"Accepted ontology delta is invalid: {path.name}.")
    return value


def _typed_binding_from_ref(
    interaction_root: Path,
    record_ref: str,
    *,
    producer: str,
) -> TypedContextBinding:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[:2] != ("products", "grounding")
    ):
        raise GroundingContractError("Typed context ref leaves products/grounding.")
    path = (interaction_root / relative).resolve()
    try:
        path.relative_to(interaction_root)
    except ValueError as exc:
        raise GroundingContractError("Typed context ref leaves its interaction.") from exc
    try:
        source = path.read_bytes()
        record = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(
            f"Typed context record could not be read: {record_ref}."
        ) from exc
    if not isinstance(record, Mapping):
        raise GroundingContractError(f"Typed context record is invalid: {record_ref}.")
    record_type = _required_symbol(record.get("record_type"), "typed record_type")
    if "schema_version" in record:
        raise GroundingContractError("Unsupported typed record fields. Start a fresh interaction.")
    embedded_evidence_stale = False
    try:
        embedded_hash_ref_count = _validate_embedded_hash_refs(
            record,
            interaction_root,
        )
    except _EmbeddedEvidenceStateError:
        embedded_evidence_stale = True
        embedded_hash_ref_count = 1
    if (
        record_type
        in {
            "RobotFramePoseRecord",
            "RobotFrameLocationRecord",
        }
        and embedded_hash_ref_count < 1
    ):
        raise GroundingContractError(
            "RobotFramePoseRecord requires a pinned source-evidence chain."
        )
    record_producer = _required_symbol(record.get("producer"), "typed producer")
    if record_producer != producer:
        raise GroundingContractError(f"Typed context producer mismatch for {record_ref}.")
    status = "stale" if embedded_evidence_stale else _binding_status(record_type, record)
    if status not in _BINDING_STATUSES:
        raise GroundingContractError(f"Typed context status is invalid: {record_ref}.")
    evidence_refs = _record_evidence_refs(record)
    provenance_value = record.get("provenance", {})
    provenance = dict(provenance_value) if isinstance(provenance_value, Mapping) else {}
    return TypedContextBinding(
        output_symbol=record_type,
        subject_role=_optional_symbol(record.get("subject_role"), "subject_role")
        or "product_context",
        record_type=record_type,
        record_ref=record_ref,
        record_sha256=hashlib.sha256(source).hexdigest(),
        status=status,
        frame=_record_frame(record),
        observed_at_ns=_record_observed_at_ns(record),
        valid_from_ns=_record_optional_integer(record, "valid_from_ns"),
        valid_until_ns=_record_optional_integer(record, "valid_until_ns"),
        producer=record_producer,
        evidence_refs=evidence_refs,
        provenance=provenance,
    )


def _binding_status(record_type: str, record: Mapping[str, object]) -> str:
    if record_type == "DocumentOverviewRecord":
        overview = record.get("overview")
        return "accepted" if isinstance(overview, Mapping) else "rejected"
    if record_type == "DocumentEvidenceRecord":
        observations = record.get("observations")
        return "accepted" if isinstance(observations, list) else "rejected"
    if record_type == "DocumentSourceIndexRecord":
        return _document_source_index_binding_status(record)
    if record_type == "DocumentQueryRecord":
        return _document_query_binding_status(record)
    if record_type == "CADSizeCorrespondenceRecord":
        return _cad_size_measurement_binding_status(record)
    if record_type == "CandidateSpatialRelationRecord":
        return _candidate_spatial_relation_binding_status(record)
    if record_type == "CADPoseEstimationRecord":
        return str(record.get("pose"))
    if record_type == "RobotFramePoseRecord":
        return str(record.get("robot_frame_conversion"))
    if record_type == "RobotFrameLocationRecord":
        return str(record.get("robot_frame_conversion"))
    if record_type in {
        "CADMeshRecord",
        "ColoredPointCloudSetRecord",
        "ObservationCandidateReview",
        "RGBDSegmentationRecord",
        "CameraToRobotCalibrationRecord",
    }:
        return "accepted"
    status = record.get("status")
    return str(status) if isinstance(status, str) else "unavailable"


def _document_source_index_binding_status(record: Mapping[str, object]) -> str:
    source_index = record.get("source_index")
    return (
        "accepted"
        if record.get("status") == "accepted"
        and isinstance(source_index, Mapping)
        and _record_fingerprint_is_valid(source_index)
        and _record_fingerprint_is_valid(record)
        else "rejected"
    )


def _document_query_binding_status(record: Mapping[str, object]) -> str:
    question = record.get("question")
    claims = record.get("claims")
    if (
        not isinstance(question, str)
        or not question.strip()
        or record.get("question_sha256") != hashlib.sha256(question.encode("utf-8")).hexdigest()
        or not isinstance(record.get("source_index"), Mapping)
        or not isinstance(claims, list)
        or not isinstance(record.get("uncertainty"), list)
        or not _record_fingerprint_is_valid(record)
    ):
        return "rejected"
    status = str(record.get("status"))
    if (status == "supported") != bool(claims):
        return "rejected"
    return {
        "supported": "accepted",
        "contradicted": "rejected",
        "insufficient_evidence": "unavailable",
    }.get(status, "rejected")


def _cad_size_measurement_binding_status(record: Mapping[str, object]) -> str:
    """Validate a neutral all-candidate CAD measurement record."""
    expected_record_keys = {
        "record_type",
        "producer",
        "correspondence_number",
        "method",
        "parameters",
        "CAD",
        "segmentation",
        "candidate_measurements",
        "measurement",
        "CAD_correspondence",
        "location",
        "pose",
        "cross_camera_fusion",
        "fingerprint",
    }
    expected_measurement_keys = {
        "observation_handle",
        "camera_id",
        "camera_order",
        "frame",
        "candidate_handle",
        "candidate_id",
        "point_count",
        "measurement_status",
        "observed_dimensions_m",
        "dimension_errors",
        "mean_dimension_error",
        "within_size_tolerance",
        "candidate_center_m",
    }
    measurements = record.get("candidate_measurements")
    if (
        set(record) != expected_record_keys
        or record.get("record_type") != "CADSizeCorrespondenceRecord"
        or record.get("producer") != "rgb_d_cad_grounding"
        or record.get("measurement") != "accepted"
        or record.get("CAD_correspondence") != "not_evaluated"
        or record.get("location") != "not_evaluated"
        or record.get("pose") != "not_evaluated"
        or record.get("cross_camera_fusion") != "not_evaluated"
        or not isinstance(measurements, list)
        or not _record_fingerprint_is_valid(record)
    ):
        return "rejected"
    for item in measurements:
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_measurement_keys
            or item.get("measurement_status")
            not in {"measured", "partial_visibility", "unreliable"}
            or not isinstance(item.get("observation_handle"), str)
            or not isinstance(item.get("candidate_handle"), str)
            or not isinstance(item.get("within_size_tolerance"), bool)
            or not _finite_vector(item.get("candidate_center_m"), 3)
        ):
            return "rejected"
    return "accepted"


def _candidate_spatial_relation_binding_status(
    record: Mapping[str, object],
) -> str:
    candidate_count = record.get("candidate_count")
    candidates = record.get("candidates")
    pairwise = record.get("pairwise_measurements")
    collinearity = record.get("collinearity_measurements")
    if (
        record.get("status") == "measured"
        and isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count >= 2
        and isinstance(record.get("segmentation"), Mapping)
        and isinstance(candidates, list)
        and len(candidates) == candidate_count
        and isinstance(pairwise, list)
        and len(pairwise) == candidate_count * (candidate_count - 1) // 2
        and isinstance(collinearity, list)
        and len(collinearity)
        == candidate_count * (candidate_count - 1) * (candidate_count - 2) // 6
        and _record_fingerprint_is_valid(record)
    ):
        return "accepted"
    return "rejected"


def _record_fingerprint_is_valid(record: Mapping[str, object]) -> bool:
    """Check a provider-owned record fingerprint without repairing the record."""
    payload = dict(record)
    fingerprint = payload.pop("fingerprint", None)
    return bool(_is_sha256_value(fingerprint) and _fingerprint(payload) == fingerprint)


def _record_frame(record: Mapping[str, object]) -> str | None:
    for key in ("coordinate_frame", "target_frame", "source_frame"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _record_observed_at_ns(record: Mapping[str, object]) -> int | None:
    value = record.get("observation_timestamp_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    cameras = record.get("cameras")
    if isinstance(cameras, list):
        timestamps = [
            camera.get("timestamp_ns")
            for camera in cameras
            if isinstance(camera, Mapping)
            and isinstance(camera.get("timestamp_ns"), int)
            and not isinstance(camera.get("timestamp_ns"), bool)
        ]
        if timestamps:
            return max(int(item) for item in timestamps)
    return None


def _record_optional_integer(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _record_evidence_refs(record: Mapping[str, object]) -> tuple[str, ...]:
    value = record.get("evidence_refs")
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    refs: list[str] = []
    for key in ("document_context_ref", "observation_ref"):
        item = record.get(key)
        if isinstance(item, str) and item:
            refs.append(item)
    cad = record.get("CAD")
    if isinstance(cad, Mapping):
        item = cad.get("context_ref")
        if isinstance(item, str) and item:
            refs.append(item)
    return tuple(refs)


def _validate_embedded_hash_refs(value: object, interaction_root: Path) -> int:
    """Validate nested local hash references and return their exact count."""
    count = 0
    if isinstance(value, Mapping):
        ref = value.get("ref")
        sha256 = value.get("sha256")
        if isinstance(ref, str) or isinstance(sha256, str):
            if not isinstance(ref, str) or not isinstance(sha256, str):
                raise GroundingContractError("Typed record artifact hash reference is incomplete.")
            relative = Path(ref)
            if relative.is_absolute() or ".." in relative.parts:
                raise GroundingContractError("Typed record artifact ref is invalid.")
            path = (interaction_root / relative).resolve()
            try:
                path.relative_to(interaction_root)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise _EmbeddedEvidenceStateError(
                    f"Typed record artifact could not be validated: {ref}."
                ) from exc
            except ValueError as exc:
                raise GroundingContractError(
                    f"Typed record artifact ref is invalid: {ref}."
                ) from exc
            if digest != sha256:
                raise _EmbeddedEvidenceStateError(
                    f"Typed record artifact hash does not match: {ref}."
                )
            count += 1
        for item in value.values():
            count += _validate_embedded_hash_refs(item, interaction_root)
    elif isinstance(value, list):
        for item in value:
            count += _validate_embedded_hash_refs(item, interaction_root)
    return count


def _abox_assertion_records(abox: ABoxSnapshot) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subject, predicate, value in sorted(
        abox.graph, key=lambda triple: tuple(str(node) for node in triple)
    ):
        if isinstance(value, URIRef):
            object_record: dict[str, object] = {"kind": "iri", "value": str(value)}
        elif isinstance(value, Literal):
            object_record = {
                "kind": "literal",
                "value": value.toPython(),
                "datatype": None if value.datatype is None else str(value.datatype),
                "language": value.language,
            }
        else:
            object_record = {"kind": "node", "value": str(value)}
        records.append(
            {
                "subject": str(subject),
                "predicate": str(predicate),
                "object": object_record,
            }
        )
    return records


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(f"{label} could not be read.") from exc
    if not isinstance(value, Mapping):
        raise GroundingContractError(f"{label} must be an object.")
    return value


def _completion_ref_path(
    interaction_root: Path,
    record_ref: str,
    *,
    prefix: tuple[str, ...] | None,
) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or (prefix is not None and relative.parts[: len(prefix)] != prefix)
    ):
        raise GroundingContractError("Grounding completion record ref is outside its authority.")
    path = (interaction_root / relative).resolve()
    try:
        path.relative_to(interaction_root)
    except ValueError as exc:
        raise GroundingContractError("Grounding completion record ref leaves its interaction.") from exc
    return path


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GroundingContractError("Pinned grounding completion record is unavailable.") from exc


def _native_assignment_delta_ref(
    root: Path,
    *,
    resource_selection_ref: str,
    selected_resource_iri: str,
) -> str:
    """Find the one host assignment delta pinned to the native selection."""
    matches: list[Path] = []
    for path in sorted((root / "products/grounding/ontology").glob("delta_*.json")):
        delta = _read_json_mapping(path, "resource assignment delta")
        if delta.get("producer") != "resource_grounding_host":
            continue
        assertions = delta.get("assertions")
        if not isinstance(assertions, list):
            continue
        has_selected_resource = any(
            isinstance(assertion, Mapping)
            and _iri_local_name(str(assertion.get("predicate"))) == "runsOnResource"
            and isinstance(assertion.get("object"), Mapping)
            and assertion["object"].get("value") == selected_resource_iri
            and assertion.get("evidence_refs") == [resource_selection_ref]
            for assertion in assertions
        )
        if has_selected_resource:
            matches.append(path)
    if len(matches) != 1:
        raise GroundingContractError(
            "Native completion requires one exact resource assignment delta."
        )
    return matches[0].relative_to(root).as_posix()


def _native_assignment_delta_ref(
    root: Path,
    *,
    resource_selection_ref: str,
    reachability_ref: str,
    validation_ref: str,
    selected_resource_iri: str,
) -> str:
    """Find the one host delta citing the complete accepted allocation lineage."""
    expected_refs = [
        resource_selection_ref,
        reachability_ref,
        validation_ref,
    ]
    matches: list[Path] = []
    for path in sorted((root / "products/grounding/ontology").glob("delta_*.json")):
        delta = _read_json_mapping(path, "resource assignment delta")
        assertions = delta.get("assertions")
        if (
            delta.get("producer") != "resource_grounding_host"
            or not isinstance(assertions, list)
            or len(assertions) != 4
            or not all(
                isinstance(assertion, Mapping) and assertion.get("evidence_refs") == expected_refs
                for assertion in assertions
            )
        ):
            continue
        has_selected_resource = any(
            _iri_local_name(str(assertion.get("predicate"))) == "runsOnResource"
            and isinstance(assertion.get("object"), Mapping)
            and assertion["object"].get("value") == selected_resource_iri
            for assertion in assertions
            if isinstance(assertion, Mapping)
        )
        if has_selected_resource:
            matches.append(path)
    if len(matches) != 1:
        raise GroundingContractError(
            "PA allocation completion requires one exact assignment delta."
        )
    return matches[0].relative_to(root).as_posix()


def _native_source_hash(
    root: Path,
    source_ref: str,
    *,
    product_requirement: str,
) -> str:
    """Hash one citation from the native PA evidence boundary."""
    if source_ref == "requirement_0001":
        return hashlib.sha256(product_requirement.encode("utf-8")).hexdigest()
    observation_manifest = root / "products" / "observations" / source_ref / "manifest.json"
    if observation_manifest.is_file():
        return _sha256_path(observation_manifest)
    relative = Path(source_ref)
    if not relative.is_absolute() and ".." not in relative.parts:
        local_path = (root / relative).resolve()
        try:
            local_path.relative_to(root)
        except ValueError:
            local_path = Path()
        if local_path.is_file():
            return _sha256_path(local_path)
    context_ref = source_ref.split("#page=", 1)[0]
    try:
        return str(approved_document_metadata(context_ref)["source_sha256"])
    except (OSError, TypeError, ValueError):
        pass
    try:
        return hashlib.sha256(approved_cad_path(context_ref).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        raise GroundingContractError(
            f"Native completion source is not approved: {source_ref}."
        ) from exc


def _write_json_mapping_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise GroundingContractError(
            f"Grounding completion record already exists: {path.name}."
        ) from exc


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GroundingContractError(f"{label} must be an object.")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise GroundingContractError(f"{label} fields are invalid. Start a fresh interaction.")


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundingContractError(f"{label} must be a non-empty string.")
    return value


def _required_symbol(value: object, label: str) -> str:
    result = _required_string(value, label)
    if not _SAFE_SYMBOL.fullmatch(result):
        raise GroundingContractError(f"{label} contains control characters.")
    return result


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GroundingContractError(f"{label} must be a nonnegative integer.")
    return value


def _optional_nonnegative_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, label)


def _sha256_string(value: object, label: str) -> str:
    result = _required_string(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise GroundingContractError(f"{label} must be a lowercase SHA-256 value.")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GroundingContractError(f"{label} must be a list.")
    result = tuple(_required_symbol(item, f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise GroundingContractError(f"{label} must not contain duplicates.")
    return result


def _optional_symbol(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_symbol(value, label)
