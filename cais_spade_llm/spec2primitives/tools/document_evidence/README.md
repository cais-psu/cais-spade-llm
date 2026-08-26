# Document evidence tool

Phase 4.1 renders all six pages of the exact approved NIST PDF, combines the
ordered page images with extracted text, and invokes `DocumentVisionRuntime`
once. `OpenAIDocumentVisionRuntime` is the production adapter and uses the model
settings in `../../config/model_runtime.json`.

The tool returns an untrusted generic triple delta. Only the shared ABox
validator can accept it. Uncertainty and unresolved evidence needs stay outside
the RDF graph. The separate UI diagnostic creates its own ABox and never calls
ProductAgent or assesses context completion.
