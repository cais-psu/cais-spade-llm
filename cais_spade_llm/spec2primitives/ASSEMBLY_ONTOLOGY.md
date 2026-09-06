# Assembly ontology and Phase 4 grounding

PA grounds instances under the supplied PPR TBox. This change preserves its symbols and pairwise cardinality.

## Fixed ontology terms

| Term | Meaning |
| --- | --- |
| `Assembly` | subclass of `product` |
| `Part` | subclass of `product` |
| `AssemblyFeature` | subclass of `feature` |
| `AssemblyFeatureAssociation` | subclass of `feature` |
| `hasPart` | `Assembly` to `product` |
| `hasAssemblyFeature` | `product` to `AssemblyFeature` |
| `hasAssemblyFeatureAssociation` | `Assembly` to association |
| `relatesAssemblyFeature` | association to exactly two `AssemblyFeature` endpoints |

Existing `hascurrentstate` and `hasdesiredstate` links identify which state a relationship describes. Several relationships use separate pairwise individuals; `hasPart` may link a `Part` or another `Assembly`. No product-specific target or preferred arm is encoded here.

## Current and desired example

Suppose the requirement and clarification specify a gear over a shaft. This is an illustrative representation, not an answer for a saved scene:

| Field | Evidence-dependent content |
| --- | --- |
| `current_state.statement` | Current product condition, including uncertain contact/attachment |
| `current_state.state_values` | Supported gear and shaft observations |
| `desired_state.statement` | Required assembled gear–shaft condition |
| `desired_state.state_values` | Supported observed destination reference, including the shaft |
| association `state_names` | `["desired_state"]` if the relationship is intended but not observed as completed |
| endpoint bindings | Both may refer to `current_state` observations |

The shaft's observed location can anchor the destination without claiming assembly is already complete or specifying the final insertion pose. A figure may show intended mating; it cannot prove current installation. Several shaft-shaped candidates may share a type. A smaller dimension error does not establish the correct task role. Distinguishing layout/relationship evidence or supported interchangeability is required; otherwise the role remains unresolved.

Each state value needs evidence for membership in that state. The moving gear's current location does not belong in `desired_state.state_values` solely as an identity reference. An observation may support both states when evidence supports its role in each. PA interprets that role without requiring a final insertion pose or host-selected value names. Deterministic checks verify references and required planning inputs; they do not establish semantic role correctness.

## Proposal 12

One overall `target_feature` contains the required process, both cited statements and typed `state_values`. For assembly, `assembly_feature_association` is a list with elements of this form:

```json
{
  "assembly": {"name": "<exact identity>", "evidence_refs": ["<source>"]},
  "state_names": ["desired_state"],
  "assembly_features": [
    {
      "name": "<first feature>",
      "owner": {"name": "<first owner>", "type": "Part", "evidence_refs": ["<source>"]},
      "state_name": "current_state",
      "state_value_name": "<existing observed value>",
      "evidence_refs": ["<source>"]
    },
    {
      "name": "<second feature>",
      "owner": {"name": "<second owner>", "type": "Assembly", "evidence_refs": ["<source>"]},
      "state_name": null,
      "state_value_name": null,
      "evidence_refs": ["<source>"]
    }
  ],
  "evidence_refs": ["<relationship source>"]
}
```

Placeholders are not runtime enums. Bound endpoints name valid coordinate-bearing state values. Both fields must be null for an unbound endpoint. `state_names` contains current, desired, or both without duplicates. PA assigns membership independently of endpoint bindings.

PA supplies names/owners/relationships/evidence. The host reuses owners only by exact identity and compiles each association under its owning `Assembly`. Assertion count varies with relationship count, membership and owner reuse. No normalization or approximate merge occurs.

Zero relationships are structurally representable. PA is responsible for requirement coverage, state meaning, endpoint identities and task roles. The host checks structure, exact bindings, evidence integrity and the planning consumer's required locations before commit. Missing required locations and unissued references return generic feedback for PA correction within 24 evidence operations and six proposals per invocation. These checks do not prove whole-goal completeness or semantic correctness.

## Assignment and images

Arm assignment adds the existing four specification/processExecution/type/process/resource links. PA checks every capable arm and selects one with accepted live MoveIt reachability for every bound current and destination coordinate reference. Planning uses the configured robot planning profile without starting or contacting a RobotAgent. Multiple relationships do not force one Cartesian pair. Exact duplicate locations are checked once per state; non-coordinate values remain semantic evidence.

Unresolved planning inputs prevent target commit; invalid evidence or failed assignment prevents grounding completion. Rejected proposals remain inspectable. MoveIt position planning does not establish grasp, insertion or physical assembly success. Phase 5 starts with selected-RA context and primitive composition.

The UI includes observed endpoint images beside each state's relationships. Desired images are labeled assembly references. Cited uncertainty remains visible after reload; qualified attachment never silently becomes verified installation.

Accepted proposals pin their pre-commit context and complete evidence dependencies in `grounding_evidence`. Incompatible saved contracts fail reload with “Start a fresh interaction”. Existing records remain unchanged and cannot start new RA work under an earlier contract. See [schemas](schemas/README.md) and [bias validation](BIAS_VALIDATION.md).
