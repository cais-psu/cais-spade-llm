# Cases

Case descriptions supply requirements and approved evidence references, not runtime answers. The current production authority contains `assembly`, `xarm6`, and `ur5e`; this is not runtime process discovery.

A case may require several pairwise assembly relationships under one `target_feature`. Current observations and desired relationship membership are separate. PA determines task roles, destination references, attachment claims and whole-goal completeness from evidence; host validation checks structure and provenance before PA assigns a capable reachable arm.

Use fresh interaction roots and frozen configurations for experiments. Vary candidate order/layout, ambiguous identities, missing/contradictory evidence, and arm reachability as described in [BIAS_VALIDATION.md](../BIAS_VALIDATION.md). Approved document/CAD names are legitimate evidence; expected candidates, simulator poses, evaluator labels and completed primitive recipes cannot be supplied to recognition.

Finalized predictions and evaluator-only outcomes remain separate under `evaluations/`. Do not silently relabel an older run as grounded under current proposal/evidence/completion checks.

Primitive-composition cases must separate changed coordinates from conditions requiring different operations/order. Use the held-out matrix in [COMPOSITION_EVALUATION.md](../COMPOSITION_EVALUATION.md), including custody, clearance and partial assembly where supported. Outputs are proposals with explicit gaps or a pass within the recorded rigid vertical geometry/motion scope; neither establishes physical success. Helper waypoints do not constitute an expected sequence, and a hex nut requires supported threading/contact behavior before claiming successful assembly.

An explicit assembly experiment specification may be a JSON file under this directory, selected by `config/phase5_validation.json` `validation_specification_path`. Supply approved position/axis tolerances and declare any required yaw, threading or force behavior. The specification must describe the outcome independently of RA's sequence; do not supply expected movements or simulator coordinates. The default path is null, so missing tolerances remain a reported gap.
