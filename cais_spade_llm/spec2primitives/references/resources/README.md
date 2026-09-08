# Resource references

`config/workcell_profile.json` and its referenced manifests define exact resource identities, process capabilities and MoveIt controller profiles. Keep `xarm6` and `ur5e` symbols unchanged. Robot capabilities and poses remain configuration-driven.

After deterministic product grounding, PA checks every capable arm against the same grounded current/destination locations and selects an arm with accepted reachability. Neither registry order nor an unsupported preference selects the arm. Selection and completion record live MoveIt reachability with retained MoveIt position plans and no grasp/insertion validation claim.

Resource-selection results never enter product recognition . After assignment, Phase 5 obtains only the selected RA's authoritative state and complete primitive catalog. See [catalog ownership](primitive_catalogs/README.md) and [bias experiments](../../BIAS_VALIDATION.md).

Fresh RA capture also reads the exact configured planning frame, controlled link and TCP link. These are distinct from the base frames used for advisory allocation distance. The owned measurement adapter separately captures full EE/TCP transforms and joints. Strict calculation checks frame compatibility without an implicit conversion or simulator lookup. Execution identifier resolution remains subsequent RA-adapter work; it never supplies recognition labels.
