# Tools

Controlled retrieval, observation, and future perception components belong
here. They are tools used by the Spec2Primitives workflow, not agents.

- `exact_ref_resolver.py` serves approved product document and CAD evidence.
- `observation_context.py` validates and stores fixture, replay, and live RGB-D
  bundles.
- `document_evidence/` is reserved for PDF text extraction and VLM diagram
  interpretation.
- `rgb_d_cad_grounding/gazebo_observation_provider.py` captures one fresh live
  Gazebo RGB-D bundle only when explicitly called. The directory remains
  reserved for later RGB segmentation, depth geometry, and CAD registration.

The live observation provider creates no background subscription and supplies
no observation to PA or RA. Future agent access must call the tool dynamically
through a separately authorized adapter.
