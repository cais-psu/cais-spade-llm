# Tools

PA controls four native investigation tools: approved `retrieve`, PA-authored `query_document`, all-candidate `compare_cad_size`, and same-view `analyze_candidate_layout`. They return typed evidence and geometry without choosing the product goal, candidate role or arm.

`document_evidence/` supplies deterministic ordered source indexes and bounded document questions. `rgb_d_cad_grounding/` supplies CAD/RGB-D processing, neutral segmentation/crops, morphological observation review, measurements and calibrated location conversion.

`observation_presentation.py` pins random interaction-local view/candidate handles and presentation order. Real sensor identities, frames and canonical pointers remain internal. Exact inverse mappings preserve transforms/provenance.

After deterministic proposal validation, PA invokes required resource reachability for every grounded state location. Every capable resource must be checked against the same grounded positions. The live MoveIt backend validates position plans without executing motion.

See [bias validation](../BIAS_VALIDATION.md), [schemas](../schemas/README.md), and [tests](../tests/README.md). No tool-level success alone authorizes semantic completion or robot execution.
