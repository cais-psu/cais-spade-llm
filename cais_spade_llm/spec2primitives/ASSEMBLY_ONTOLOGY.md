# Assembly ontology and Phase 4 grounding

This is the implemented assembly profile for Spec2Primitives. It is an
OAM-informed local extension of the existing PPR TBox, not a complete import of
the NIST [Ontology for Assembly Representation](https://www.nist.gov/publications/ontology-assembly-representation).
It adds only the structure consumed by the current Phase 4 assembly path.

## TBox

The existing PPR terms remain unchanged. The assembly slice adds these exact
classes and properties:

| Term | Axiom |
| --- | --- |
| `Assembly` | subclass of `product` |
| `Part` | subclass of `product` |
| `AssemblyFeature` | subclass of `feature` |
| `AssemblyFeatureAssociation` | subclass of `feature` |
| `hasPart` | `Assembly` to `product` |
| `hasAssemblyFeature` | `product` to `AssemblyFeature` |
| `hasAssemblyFeatureAssociation` | `Assembly` to `AssemblyFeatureAssociation` |
| `relatesAssemblyFeature` | `AssemblyFeatureAssociation` to `AssemblyFeature` |

`AssemblyFeatureAssociation` has exactly two
`relatesAssemblyFeature` endpoints of type `AssemblyFeature`. `hasPart` already
allows either a `Part` or another `Assembly` because both are `product`; a
separate `hasSubassembly` property is unnecessary. There is no `Artifact`,
`ArtifactAssociation`, `concerns`, or `changing_assembly_feature` term.

This TBox supplies allowed structure. It does not contain `Medium Gear`, USB,
hex nut, chair, cable, or any other product-specific answer.

## Autonomous and deterministic responsibilities

| ProductAgent authors autonomously | Host and validator do deterministically |
| --- | --- |
| which approved documents, CAD, and observations to retrieve | expose only approved, hash-pinned evidence through opaque handles |
| the `required_process` | check that the process is configured |
| current and desired statements | check the response shape and direct citations |
| state-value names and exact typed-record paths | resolve each JSON Pointer and verify its record hash |
| the `Assembly`, both `AssemblyFeature` endpoints, and their owners | create interaction-local IRIs and compile the exact RDF assertions |
| which endpoint binds to `current_state` and which binds to `desired_state` | require one endpoint per state and exactly two distinct endpoints |
| the exact coordinate-bearing evidence for both endpoints | verify that each bound value is an allocatable location |
| the capable resource to test | execute reachability checks and accept or reject the unchanged choice |

The validator does not choose a gear, shaft, port, stud, resource, image, or
desired location. It also does not repair a bad PA answer.

## End-to-end medium gear example

Requirement:

```text
assemble medium gear
```

The shaft is already installed in the assembly board. It is not the component
to pick. The document figure supplies semantic evidence that the medium gear is
to mate with that shaft. The live observation supplies the coordinate-bearing
evidence for where the medium gear is now and where the assembly board shaft is
now.

The intended PA reasoning is:

```text
current_state endpoint: medium gear bore at the observed medium gear location
desired_state endpoint: assembly board shaft at the observed shaft location
assembly action: move the medium gear so its bore mates with the installed shaft
```

A shortened PA-authored proposal looks like this:

```json
{
  "target_feature": {
    "required_process": {
      "process_iri": "https://cais-spade-llm.local/process/assembly",
      "evidence_refs": ["requirement_0001", "evidence_0001"]
    },
    "current_state": {
      "statement": {
        "text": "The medium gear is separate from the assembly board shaft.",
        "evidence_refs": ["evidence_0003"]
      },
      "state_values": [
        {
          "name": "medium_gear_location",
          "value_ref": {
            "record_ref": "<accepted RGBDSegmentationRecord>",
            "field_path": "/cameras/0/candidates/1"
          },
          "evidence_refs": ["evidence_0002", "evidence_0003"]
        }
      ]
    },
    "desired_state": {
      "statement": {
        "text": "The medium gear bore is mated with the shaft already installed in the assembly board.",
        "evidence_refs": ["evidence_0001", "evidence_0003"]
      },
      "state_values": [
        {
          "name": "assembly_board_shaft_location",
          "value_ref": {
            "record_ref": "<accepted RGBDSegmentationRecord>",
            "field_path": "/cameras/1/candidates/0"
          },
          "evidence_refs": ["evidence_0001", "evidence_0003"]
        }
      ]
    },
    "assembly_feature_association": {
      "assembly": {
        "name": "medium gear assembly",
        "evidence_refs": ["requirement_0001", "evidence_0001"]
      },
      "assembly_features": [
        {
          "name": "medium gear bore",
          "owner": {
            "name": "medium gear",
            "type": "Part",
            "evidence_refs": ["evidence_0001", "evidence_0002"]
          },
          "state_name": "current_state",
          "state_value_name": "medium_gear_location",
          "evidence_refs": ["evidence_0001", "evidence_0002"]
        },
        {
          "name": "assembly board shaft",
          "owner": {
            "name": "assembly board",
            "type": "Assembly",
            "evidence_refs": ["evidence_0001", "evidence_0003"]
          },
          "state_name": "desired_state",
          "state_value_name": "assembly_board_shaft_location",
          "evidence_refs": ["evidence_0001", "evidence_0003"]
        }
      ],
      "evidence_refs": ["evidence_0001", "evidence_0003"]
    }
  }
}
```

The host then creates `feature_0001` as an
`AssemblyFeatureAssociation`, `assembly_0001`, two owners, and two
`AssemblyFeature` individuals. It connects the association to its two endpoints
and retains the existing `currentstate_0001` and `desiredstate_0001` links. For
an assembly proposal, this is exactly 19 proposal assertions. The later
resource assignment still contributes the existing four `processExecution`
assertions.

The allocation call does not independently choose current and desired
locations. It receives the exact two state-location handles derived from the
accepted PA proposal and lets PA choose only a capable resource. This prevents
the semantic proposal and reachability request from silently referring to
different locations.

## Current and desired images

Images are not forced under current or desired state by a product rule. PA
selects the exact coordinate-bearing record for each state value. After that
choice is accepted, deterministic UI code resolves the corresponding immutable
record and displays its available annotation or crop under that selected role.

Therefore, image placement is autonomous at the semantic decision boundary and
deterministic only for provenance-preserving rendering. Candidate order, camera
name, provider metadata, and a hard-coded medium-gear rule do not assign the
role.

## General assembly validation

The same structural validation applies without product-specific predicates:

| Requirement | `current_state` endpoint | `desired_state` endpoint |
| --- | --- | --- |
| insert USB plug | USB plug mating face | USB port mating opening |
| install hex nut | hex nut threaded hole | stud external thread |
| connect cable device | cable connector mating face | device socket |
| attach chair leg | chair-leg mounting face | seat mounting interface |

Names and evidence differ, but the validator checks the same facts: one
`AssemblyFeatureAssociation`, two distinct `AssemblyFeature` endpoints, distinct
owners, one endpoint bound to each state, accepted coordinate-bearing evidence,
and exact proposal/allocation agreement. An assembly that does not contain a
gear or shaft does not need either term.

A complicated product is handled as one evidence-backed target association per
Phase 4 interaction. A cable assembly or chair may require several such target
associations and an ordered downstream program. Accepting multiple associations
inside one Phase 4 proposal is intentionally deferred; the current contract
must not imply that a one-association completion represents an entire complex
product.

## What this fixes and what it does not

This change addresses the observed semantic error by requiring an assembly PA
proposal to identify the two mating endpoints and to bind the desired endpoint
to the assembly board shaft location. A proposal that omits this structure,
uses non-location evidence, swaps or duplicates state roles, or changes the
location during allocation now fails closed.

It does not guarantee that PA will choose the correct evidence, and the
validator does not substitute the expected answer. Correct behavior still
depends on adequate document and observation evidence. It also does not make an
unreachable target reachable: after the semantic fix, a run may correctly end
with `no_reachable_resource` when no one capable resource can reach both the
selected medium gear location and the selected assembly board shaft location.

New runs persist `OntologyGroundingProposal` v10 and
`PAContextGroundingCompletion` v8. Completion v7 remains readable for audit but
cannot enter the current Phase 5 handoff because it predates the assembly
association and proposal-bound location contract; rerun Phase 4 to create v8.
