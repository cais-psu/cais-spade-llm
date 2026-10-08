# Context-free PPR atomic propositions

The authoritative AP is structured data with product, process, resource, and one
typed state or event condition. The existing prefixes remain ap_state and
ap_event. There is no AP context field and no slash-encoded identity.

~~~json
{
  "kind": "ap_state",
  "product": "*",
  "process": "*",
  "resource": "ur5e-1",
  "state": {"symbol": "any", "arguments": {"region": "assembly_board-v1"}}
}
~~~

This is the structured form of ap_state(*, *, ur5e-1, any@assembly_board-v1).
The state symbol any imposes no discrete-state filter. The region qualifier is
evaluated for the configured resource geometry, including tool and carried part,
at every modeled observation. Destination parameters or task names do not prove
physical presence. Release changes carried-part association but does not clear
the robot's region condition. A deposited part is no longer part of its carrier.

Resource wildcard authoring expands against the complete configured scene before
monitor execution. build_mutex_specification accepts a typed resource "*" schema
and produces internal pair slots. Pairwise mutex instances contain two concrete,
distinct resource APs and G !(ap001 & ap002). Internal first/second references
are binding slots, not separate spatial predicates. Every unordered resource
pair is covered, including fixed equipment. AP labels remain local to each rule.

Typed conditions also cover part-region entry, exact processCompleted records,
receiving-region entry, inventory, task events, and resource state values.
Part entry is a false-to-true physical transition; an initially present part and
a custody-only change do not fabricate entry. Process completion requires an
explicit complete ledger and exact process/target or process/result record.

Primitive executions determine the observation trace, then typed predicates
produce AP values, then the supplied temporal formula advances its monitor.
Intermediate motion observations and accepted monitor history must be retained.
Unknown effects or missing evidence remain unavailable, never false.

ppr_ap.py validates definitions and derives canonical JSON keys without renaming
manufacturing symbols. Old slash descriptors and AP context fields fail with a
recompile requirement. Version 2 catalogs and regenerated saved artifacts replace
the previous AP implementation.
