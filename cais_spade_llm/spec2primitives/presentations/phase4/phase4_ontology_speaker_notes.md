# Why ontology matters — PPR and the assembly extension

This companion to the Phase 4 overview has **9 main slides timed to 10 minutes** and **6 optional technical backup slides**. Slides 10–15 are hidden in the PowerPoint’s normal Slide Show. Unhide them for a longer presentation; the PDF includes all 15 pages.

- [Editable PowerPoint with presenter notes](phase4_ontology_deep_dive.pptx)
- [All slides as PDF](phase4_ontology_deep_dive.pdf)
- [Original Phase 4 overview](phase4_technical_overview.pptx)

## Direct answer and presentation thesis

**The shared PPR semantic model is useful and already participates in the current runtime contract. OWL/RDF is a representation choice, not a theoretical prerequisite for assembly planning.** A typed JSON representation with the same classes, identities, relationships and validators could preserve much of that model. Comparing it with RDF tests representation and integration choices; it does not remove the underlying semantics.

The strongest justification is specific to this implementation:

1. **A shared task model:** `specification —defines→ feature ←realizes— process` connects the exact requirement to the target of the assigned process. PA and RA use the same explicit links.
2. **Assembly structure:** `Assembly`, `Part`, `AssemblyFeature` and `AssemblyFeatureAssociation` represent membership, ownership and pairwise relationships, including a relationship’s current/desired state membership.
3. **Controlled assignment:** configured `resource —capableOf→ process` facts identify candidates; `processExecution` records the accepted assignment. The selected RA validates that assignment against the accepted ontology projection.
4. **Inspectable evidence links:** assertions are joined with proposals, typed records, provenance and completion hashes. That traceability is implemented by the combined contract; an OWL class hierarchy alone does not supply it.
5. **Reuse of the vocabulary:** different task instances can use the same permitted types and relationships. This supports a stable shared interface; reduced maintenance cost or improved generalization has not been measured here.

A presentation-ready claim is:

> We use PPR and its assembly extension as the shared model connecting the product goal, assembly relationships, process and resource assignment. PA grounds task instances from evidence; the host validates the resulting contracts before downstream use.

Avoid presenting ontology as automatically verifying perception, choosing the correct shaft, proving task completeness, generating `primitive_steps`, certifying physical feasibility, or demonstrating better accuracy than an equivalent structured representation.

OWL semantics and runtime validation have different roles. OWL does not make missing required input fields a syntax error; the project adds explicit Python checks. [W3C OWL 2 Primer](https://www.w3.org/TR/owl2-primer/). [SHACL](https://www.w3.org/TR/shacl/#validation) is a separate graph-validation standard; these inspected project validators are custom Python contracts.

## Timing and how to use the slides

| Slide | Topic | Time | Cumulative |
| --- | --- | --- | --- |
| 1 | Is an ontology needed here? | 60 s | 1:00 |
| 2 | PPR connects Product, Process and Resource | 60 s | 2:00 |
| 3 | The core PPR graph links a requirement to resources | 90 s | 3:30 |
| 4 | Schema, facts and evidence have different jobs | 60 s | 4:30 |
| 5 | The assembly extension adds four classes | 60 s | 5:30 |
| 6 | Four properties add ownership and pairwise relations | 75 s | 6:45 |
| 7 | The medium-gear goal becomes concrete instances | 75 s | 8:00 |
| 8 | Relationship state and observation bindings are distinct | 60 s | 9:00 |
| 9 | Justify ontology through the contracts it supports | 60 s | 10:00 |
| 10 | processExecution records the selected assignment | Optional backup | — |
| 11 | OWL semantics and runtime validation are different | Optional backup | — |
| 12 | The ABox is a compact graph of a richer grounded task | Optional backup | — |
| 13 | The explicit relations support concrete questions | Optional backup | — |
| 14 | Compare semantic structure and representation separately | Optional backup | — |
| 15 | Reference: exact PPR property profiles | Optional backup | — |

The new deck stands alone as a 10-minute ontology presentation. To strengthen the **original** Phase 4 deck, use these substitutions:

- Replace original slide 2 with new slide 3 for the exact core PPR graph.
- Replace original slide 5 with new slide 1 for the ontology justification.
- Replace original slide 7 with new slide 7 for the detailed saved assembly ABox.
- Keep new slides 5, 6, 8 and 10–15 available for questions; adding them all requires a longer talk.

To preserve the original 10-minute timing, keep each replacement in its original 60-second slot and shorten the narration for new slides 3 and 7. Using their full companion scripts would add 45 seconds.

## Diagram conventions

- Schema diagrams show **classes** and property domain/range or subclass relationships. They do not assert links between OWL classes as if the classes were product instances.
- The medium-gear diagram shows **saved ABox individuals**. Labels beneath their identifiers are exact names from the PA proposal. The saved ABox does not contain those names as `rdfs:label` triples.
- `ctx:`, `ppr:`, `process:` and `resource:` are the prefixes used in the saved Turtle file. Full prefix declarations are provided in the cited ABox; identifiers are preserved exactly.
- `current_state`, `desired_state`, `state_names`, `state_name` and `state_value_name` are proposal/record fields. `hascurrentstate`, `hasdesiredstate`, `currentstate_0001` and `desiredstate_0001` are distinct ontology symbols or generated individual suffixes.
- Named interfaces and their ownership are PA interpretations backed by cited records. The saved example is not an independently scored semantic or physical assembly result.


## Slide 1 — Is an ontology needed here?

**Time:** 60 seconds.

**Say:**

My answer separates the shared semantic model from its encoding. The current implementation depends on PPR: PA grounds a target under its vocabulary, the host validates its assertions, and RA context reconstruction checks the same assignment links. That is an implemented runtime contract. OWL and RDF are not the only way to encode it. Typed JSON could preserve the same identities, relations and checks, and would therefore retain much of the same semantic model. The reason to retain PPR here is its explicit account of the product goal, assembly structure, process and resource assignment across the PA-to-RA boundary. The OWL schema makes those terms and profiles separately inspectable. This is an architectural justification. Better recognition, fewer model errors and improved assembly success remain claims that need a fair experiment.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Compile the overall target feature](../../agents/pa/ontology_grounding.py) — line 1049
- [Validate the interaction semantic bridge](../../agents/pa/product_context.py) — line 951
- [Validate the ontology projection for the selected RA](../../agents/ra/composition_context.py) — line 387

## Slide 2 — PPR connects Product, Process and Resource

**Time:** 60 seconds.

**Say:**

PPR organizes the problem into Product, Process and Resource views. The product view describes what is involved and the feature and state that the specification requires. In this case, the product-level content includes the medium gear, the gear plate assembly and their desired relationship. The process view describes the high-level activity that realizes a feature, here assembly. It also has processExecution for recording a particular assignment. The resource view identifies which configured resources are broadly capable of the process, here xarm6 and ur5e. Broad capability is only a candidate-discovery fact; MoveIt checks are still required for this task's grounded positions. The supplied vocabulary also declares capability as a class, but the active workcell graph uses resource capableOf process directly. The class diagram should preserve that implementation instead of inventing an intermediate capability individual or putting robot primitives into the PPR capability graph.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Workcell graph projection](../../ontology/workcell.py) — line 247
- [Configured process/resource capability authority](../../ontology/workcell.py) — line 112
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)

## Slide 3 — The core PPR graph links a requirement to resources

**Time:** 90 seconds.

**Say:**

Read the arrows in their exact direction. A specification defines a feature. A process realizes that feature. The resource is capableOf the process. This gives a common join from the requirement to its feature, the configured process, and the resources that may be considered. The feature separately hascurrentstate and hasdesiredstate. Those property names and their capitalization are fixed symbols in this project. Each is declared functional. The target proposal supplies both states, while the runtime explicitly checks its required inputs. The model should not draw a process-to-product realizes link, because this supplied TBox says process to feature. Nor should it add a feature-to-resource capableOf edge. The later assignment uses a separate processExecution node. Notice also that the base vocabulary declares product, but this active bridge has no generic product-to-feature ownership property. The assembly extension supplies that ownership through hasAssemblyFeature, allowing the bridge to connect to concrete part and assembly structure. This slide is a schema view: boxes are classes, and arrows show property domain and range.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Compile the overall target feature](../../agents/pa/ontology_grounding.py) — line 1049
- [Validate the interaction semantic bridge](../../agents/pa/product_context.py) — line 951
- [Workcell graph projection](../../ontology/workcell.py) — line 247

## Slide 4 — Schema, facts and evidence have different jobs

**Time:** 60 seconds.

**Say:**

The implementation is a hybrid of graph structure and linked records. The supplied TBox contains the generic vocabulary, class hierarchy, property profiles and pairwise cardinality. It contains no medium-gear answer, resource instance or primitive recipe. The interaction ABox contains the specification, target feature, state individuals, assembly structure and eventual selected assignment for this run. A separate configured Workcell ABox supplies process and resource facts. PA cannot invent new resource capability facts in its interaction graph. Detailed state statements, exact names, observation pointers, coordinates and uncertainty are kept in the proposal and typed evidence records. Assertion provenance and completion records retain their evidence references and hashes. Therefore a bare Turtle export does not carry all of the grounding evidence. The overall contract combines the graph with those records. This separation lets us keep a compact relational model while preserving the original measurement artifacts and the authority that supplied them.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Workcell graph projection](../../ontology/workcell.py) — line 247
- [Proposal, evidence and assignment records](../../schemas/README.md) — line 19

## Slide 5 — The assembly extension adds four classes

**Time:** 60 seconds.

**Say:**

The extension adds four classes while preserving the PPR vocabulary. Assembly and Part are subclasses of product. AssemblyFeature and AssemblyFeatureAssociation are subclasses of feature. An AssemblyFeature represents an interface or feature belonging to a product, such as the medium gear central bore. An AssemblyFeatureAssociation gives a relationship its own identity, relating exactly two AssemblyFeature endpoints. Making the association a feature is useful because the existing current-state and desired-state vocabulary can describe whether that relationship is observed or intended. Part and Assembly remain distinct supplied types, while hasPart ranges over product so an Assembly can contain another Assembly. The extension is deliberately small: it represents ownership and pairwise relationships. It does not encode contact mechanics, insertion tolerances, primitive order or a product-specific recipe. Those extra claims require their own evidence or operational model.

**Source checks:**

- [Assembly classes and cardinality](../../ontology/spec2primitives_ppr_tbox.owl) — line 19
- [Assembly property domains and ranges](../../ontology/spec2primitives_ppr_tbox.owl) — line 86
- [Vocabulary, domains, ranges and cardinality regression](../../tests/test_ppr_tbox.py) — line 27
- [Assembly semantics and evidence bindings](../../ASSEMBLY_ONTOLOGY.md) — line 5

## Slide 6 — Four properties add ownership and pairwise relations

**Time:** 75 seconds.

**Say:**

The four object properties provide the structure missing from the base bridge. hasPart links an Assembly to its product members; because the range is product, a member may be a Part or another Assembly. hasAssemblyFeature links a product to an AssemblyFeature that it owns. hasAssemblyFeatureAssociation links an Assembly to a relationship individual. Finally, relatesAssemblyFeature links that association to its two feature endpoints. The qualified cardinality applies to each association, not to the number of parts in the assembly. For three different relationships, the current compiler creates three association individuals, each with two distinct endpoint identities. Exact owners and features can be reused across associations, which avoids copying a shared part merely because it participates in more than one relationship. The association also gives us a stable place to express desired-state membership and preserve relationship evidence in the supporting records. A simple membership list alone would not express which interfaces are intended to relate.

**Source checks:**

- [Assembly property domains and ranges](../../ontology/spec2primitives_ppr_tbox.owl) — line 86
- [Assembly classes and cardinality](../../ontology/spec2primitives_ppr_tbox.owl) — line 19
- [Compile separate pairwise associations and exact owners](../../agents/pa/ontology_grounding.py) — line 864
- [Validate two distinct endpoints and exact bindings](../../agents/pa/ontology_grounding.py) — line 720
- [Separate pairwise associations and state membership](../../tests/test_pa_grounding_contracts.py) — line 493
- [Owners are reused only by exact identity](../../tests/test_pa_grounding_contracts.py) — line 667

## Slide 7 — The medium-gear goal becomes concrete instances

**Time:** 75 seconds.

**Say:**

This is a structural subgraph from the same saved medium-gear interaction as the original Phase 4 presentation. The identifiers and edges come from its accepted ABox; the readable names are annotations from the saved PA proposal, not additional rdfs:label triples. assembly_0001 is named gear plate assembly. Its hasPart links include part_0001, named medium gear, and assembly_0002, named middle gear shaft on gear plate. The latter owner is typed Assembly in that saved proposal; I have kept the supplied type rather than silently changing it to Part. Each owner has its own AssemblyFeature. The two named interfaces are medium gear central bore and middle gear shaft mounting surface. A separate assemblyfeatureassociation_0001 links those endpoints and belongs to assembly_0001. This is why an association node is useful: the relationship has an identity separate from the whole target feature and separate from either component. Its desired-state linkage is shown on the next slide. The saved interpretation has not been independently scored for semantic correctness in preparing these slides.

**Source checks:**

- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Compile separate pairwise associations and exact owners](../../agents/pa/ontology_grounding.py) — line 864
- [Owners are reused only by exact identity](../../tests/test_pa_grounding_contracts.py) — line 667

## Slide 8 — Relationship state and observation bindings are distinct

**Time:** 60 seconds.

**Say:**

An intended relationship can be grounded using objects that are observed now. In the saved proposal, the association has state_names equal to desired_state, and its ABox individual hasdesiredstate desiredstate_0001. That says where the relationship belongs in the task model. Separately, each endpoint has a state_name and a state_value_name that identify supporting observation evidence. The gear endpoint binds to medium gear current location in current_state. The shaft endpoint binds to observed middle gear shaft destination reference in desired_state. Both values ultimately refer to observed candidates. The contract also permits both endpoint bindings to use current_state; the regression test explicitly exercises that case. The relationship's desired membership therefore does not assert that assembly has already happened. The shaft location provides an observed destination reference; it is not by itself the final gear pose or the end-effector target. Distinguishing these concepts is a concrete benefit of making the relationship explicit.

**Source checks:**

- [Assembly semantics and evidence bindings](../../ASSEMBLY_ONTOLOGY.md) — line 5
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Validate two distinct endpoints and exact bindings](../../agents/pa/ontology_grounding.py) — line 720
- [Endpoints can share a current-state binding](../../tests/test_pa_grounding_contracts.py) — line 640

## Slide 9 — Justify ontology through the contracts it supports

**Time:** 60 seconds.

**Say:**

The useful justification should point to runtime responsibilities. First, defines and realizes connect the requirement to the feature that the assigned process must realize. Second, the assembly extension represents membership, feature ownership and individual pairwise relationships without adding a new schema for each observed product. Third, the capability and assignment links give the host and selected RA an explicit, checkable description of who is assigned to which process for this specification. Fourth, those relations are tied to evidence through the proposal, provenance and completion records. These are implemented architectural functions, not proof of an accuracy improvement. The practical benefit is a shared representation whose connections can be validated and inspected across components. The fair alternative is a typed JSON model containing the same identities, relationships and checks. We should compare against that alternative before attributing reduced errors or better task outcomes specifically to the ontology. For this architecture I would retain the shared model and keep its evidence boundaries explicit.

**Source checks:**

- [Validate the interaction semantic bridge](../../agents/pa/product_context.py) — line 951
- [Compile separate pairwise associations and exact owners](../../agents/pa/ontology_grounding.py) — line 864
- [Configured process/resource capability authority](../../ontology/workcell.py) — line 112
- [Validate the ontology projection for the selected RA](../../agents/ra/composition_context.py) — line 387
- [Proposal, evidence and assignment records](../../schemas/README.md) — line 19

## Slide 10 — processExecution records the selected assignment

**Optional backup; outside the 10-minute main talk.**

**Say:**

This backup shows the narrow assignment vocabulary. In the saved run, specification_1 hasProcessExecution process_execution_0001. That individual is typed processExecution, runsProcess process:assembly, and runsOnResource resource:xarm6. Those are four system-authored assertions, including the rdf:type statement. They are committed only after validation of PA's selected resource and supporting evidence. The current ABox validator requires a complete current assignment with one process and one resource. The TBox marks runsProcess and runsOnResource functional, while hasProcessExecution itself is not functional. The stricter one-current-assignment policy is implemented by host code. The existence of a processExecution individual records the assignment here; it does not assert that robot motion happened. The corresponding completion records motion_executed false. Broad capableOf facts belong to configured Workcell authority and do not by themselves select xarm6.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Validate one current system-authored assignment](../../agents/pa/product_context.py) — line 1011
- [Saved Phase 4 completion](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/context_completion_0001.json)

## Slide 11 — OWL semantics and runtime validation are different

**Optional backup; outside the 10-minute main talk.**

**Say:**

OWL specifies logical meaning. Under its open-world semantics, missing information is not automatically false. A functional property limits its semantic values but does not require an explicit input field. Likewise, a cardinality axiom is not a parser check that two endpoint records were supplied. W3C distinguishes this from data validation; SHACL is a separate standard for validating graph constraints. In this project the relevant checks are explicit Python contracts: the TBox loader verifies the supplied profile, the PA proposal validator requires exactly two distinct endpoint identities, state bindings must reference accepted coordinate-bearing values, and the assignment validator requires complete links. The named-class traversal supports subclass-aware checks, but these inspected functions are not a general OWL consistency or planning reasoner. Do not credit OWL alone with rejecting every missing field, checking source hashes, establishing semantic task correctness or certifying geometry. Those responsibilities are supplied by the host, evidence tools and planning adapters.

**Source checks:**

- [W3C OWL 2 Primer: semantics and syntactic completeness](https://www.w3.org/TR/owl2-primer/)
- [W3C SHACL: separate graph validation](https://www.w3.org/TR/shacl/#validation)
- [TBox validation and named-class traversal](../../ontology/ppr_tbox.py) — line 61
- [Validate two distinct endpoints and exact bindings](../../agents/pa/ontology_grounding.py) — line 720
- [Validate one current system-authored assignment](../../agents/pa/product_context.py) — line 1011

## Slide 12 — The ABox is a compact graph of a richer grounded task

**Optional backup; outside the 10-minute main talk.**

**Say:**

This slide makes the storage boundary visible. The actual Turtle graph types feature_0001 and connects it to currentstate_0001 and desiredstate_0001. The graph does not put the detailed state description and numeric observation data into new PPR datatype properties. In fact, the supplied TBox declares no datatype properties. The saved target_feature proposal contains the desired-state statement and exact state-value names. Its value_ref selects an exact field of a typed record. In this example, the field is /cameras/3/candidates/2 in the saved segmentation record, and the corresponding records preserve the geometry and provenance. The canonical paths are shown for audit; model-facing handles and order are projected separately. The supporting evidence and hash chain are part of the grounding contract, rather than consequences of a class hierarchy. Coordinates must still be interpreted in their frame and for their supported semantic role.

**Source checks:**

- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Proposal, evidence and assignment records](../../schemas/README.md) — line 19
- [Vocabulary, domains, ranges and cardinality regression](../../tests/test_ppr_tbox.py) — line 27

## Slide 13 — The explicit relations support concrete questions

**Optional backup; outside the 10-minute main talk.**

**Say:**

These are inspection patterns over the accepted representation, rather than a claim that the current RA authoring loop exposes an interactive SPARQL service. The runtime uses targeted graph lookups, configured capability matching and accepted assertion projections. We can trace the feature defined by a specification, the process realizing that feature, the configured resources broadly capable of that process, the selected assignment, and the two interfaces linked by an assembly association. Readable endpoint names require joining the identifiers to the saved proposal. That join is a practical benefit of explicit identities: shared structure can be inspected without interpreting a long natural-language transcript again. The current RA input constructor specifically validates defines, realizes and the assignment relations before returning its ontology projection. Queryability is therefore a property of the representation; the tool interface and its supported operations should be reported separately from it.

**Source checks:**

- [Saved interaction ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl)
- [Saved PA target_feature and exact labels](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Configured process/resource capability authority](../../ontology/workcell.py) — line 112
- [Validate the ontology projection for the selected RA](../../agents/ra/composition_context.py) — line 387

## Slide 14 — Compare semantic structure and representation separately

**Optional backup; outside the 10-minute main talk.**

**Say:**

A fair study separates the value of explicit task structure from the choice of encoding. A prompt-only narrative can test the first question, but it is not an adequate sole comparator for claiming that OWL is necessary. Typed JSON that preserves the same identities, relations, evidence and validators is a comparator for the representation, not for the absence of a semantic model. Compare it with the current PPR graph plus linked records while holding the evidence, observations, model settings, budgets, candidate presentation and planning conditions fixed. Measure task-role accuracy with a separate evaluator, reference and handoff failures, correction effort, token cost and latency. Also document schema maintenance and integration effort if interoperability is part of the motivation. These are proposed measurements, not results. If equivalent JSON performs similarly, justify PPR through the shared vocabulary and integration requirements; do not assert an unmeasured accuracy advantage.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Proposal, evidence and assignment records](../../schemas/README.md) — line 19
- [Validate the ontology projection for the selected RA](../../agents/ra/composition_context.py) — line 387

## Slide 15 — Reference: exact PPR property profiles

**Optional backup; outside the 10-minute main talk.**

**Say:**

This reference table is generated directly from the supplied OWL file. It includes all twelve object properties with their exact domain and range, and whether each is declared functional. There are four functional properties: hascurrentstate, hasdesiredstate, runsProcess and runsOnResource. The four assembly properties are not functional. The pairwise restriction is a qualified cardinality on AssemblyFeatureAssociation through relatesAssemblyFeature, so the fact that the property is not functional does not remove that restriction. The supplied vocabulary has twelve classes and no declared datatype properties. Do not add commonly expected PPR properties such as a generic hasFeature unless they are explicitly added to the approved schema in a separate implementation change. This deck documents the supplied vocabulary exactly.

**Source checks:**

- [Exact supplied PPR TBox](../../ontology/spec2primitives_ppr_tbox.owl) — line 10
- [Vocabulary, domains, ranges and cardinality regression](../../tests/test_ppr_tbox.py) — line 27
- [Assembly classes and cardinality](../../ontology/spec2primitives_ppr_tbox.owl) — line 19

## Questions worth preparing for

**Is ontology necessary, or is JSON enough?** The shared semantic model is the design value. The current code consumes an OWL-defined RDF representation, so deleting that layer would require replacing its contracts and consumers. Equivalent typed JSON could implement the same restricted model and checks. JSON is a format; using JSON does not imply the absence of ontology-like semantics.

**What does the assembly extension add beyond a list of parts?** It separates component membership (`hasPart`), interface ownership (`hasAssemblyFeature`) and a relationship with its own identity (`AssemblyFeatureAssociation`). That relationship connects exactly two interfaces and can carry current or desired state membership independently of their observed locations.

**Why exactly two endpoints?** This is the approved pairwise model. Multiple relations use multiple association individuals, and exact endpoints/owners can be reused. The cardinality does not limit the whole assembly to two parts and does not encode an assembly order.

**Why is the shaft owner typed `Assembly` in the example?** That is the exact saved PA proposal: `middle gear shaft on gear plate` has type `Assembly`. The schema permits both `Part` and `Assembly` owners. The slides preserve the record rather than changing its interpretation. This is not a claim that all shafts should be classified as `Assembly`.

**Are current and desired states actually encoded in RDF?** Their typed individuals and feature/association links are encoded. Detailed state statements, names and typed value references live in the linked target proposal. Coordinates, frames and source data remain in typed evidence records.

**Does the graph itself contain evidence hashes?** The inspected contract stores evidence references and hash pins in proposal, provenance and completion records. Do not draw invented PPR hash or geometric datatype properties.

**Does `capability` connect an arm to a primitive catalog?** The class is declared, but the active Workcell ABox uses `resource capableOf process` directly. Primitive catalog details remain separate typed RA records. The schema does not entail a task-specific primitive recipe.

**Does `processExecution` mean the task has executed?** In this Phase 4 path it is the current assignment representation. Its saved completion has `motion_executed: false`. Execution acknowledgment and physical outcome evidence are separate responsibilities.

**Is a full ontology reasoner used?** The inspected path loads RDF with RDFLib, validates the supplied profile, traverses named superclass/equivalence links for type checks and enforces specific Python contracts. This presentation does not claim general OWL consistency reasoning, SHACL execution, arbitrary SPARQL execution, semantic correctness or ontology-derived planning.

**Can RA query the ontology interactively?** The representation is inspectable and the current RA context builder consumes accepted assertions. The currently inspected authoring function performs one semantic RA decision without a model-driven evidence conversation. Residual `query_ontology` action parsing is not evidence that the active authoring path offers that tool. The backup query slide presents inspection patterns, not a claim about an active query API.

**What experiment would establish the benefit?** Compare the same PPR semantics and checks encoded in typed JSON versus the RDF graph, while controlling evidence, model and planning conditions. Separately compare explicit structure with a narrative-only input. Report role accuracy, contract errors, corrections, cost and integration effort. No such comparison was run for these slides.

## Outcome

Added [phase4_ontology_deep_dive.pptx](phase4_ontology_deep_dive.pptx), [phase4_ontology_deep_dive.pdf](phase4_ontology_deep_dive.pdf) and [phase4_ontology_speaker_notes.md](phase4_ontology_speaker_notes.md). This is presentation/documentation work with no runtime entry-point change.

## Process flow

Document-reading flow: `supplied PPR TBox → schema validator → PA proposal/assembly compiler → saved ABox + proposal → Workcell and assignment contracts → RA ontology projection`.

## Read these locations in order

1. [Supplied TBox](../../ontology/spec2primitives_ppr_tbox.owl) — lines 10, 19, 42 and 86: inspect the exact classes, extension, direction of each property and pairwise cardinality.
2. [load_ppr_tbox and TBoxSnapshot](../../ontology/ppr_tbox.py) — lines 107 and 61: a schema graph enters; a fingerprinted profile and named-class lookup surface come out.
3. [_validated_assembly_feature_association and _compile_associations](../../agents/pa/ontology_grounding.py) — lines 720 and 864: a PA relationship proposal enters; validated endpoint bindings and exact-owner ABox assertions come out. [_compile_proposal_delta](../../agents/pa/ontology_grounding.py) — line 1049: inspect the root feature, state individuals, `defines` and `realizes` links.
4. [_validate_semantic_bridge and _validate_process_execution](../../agents/pa/product_context.py) — lines 951 and 1011: the interaction graph enters; target/assignment consistency is checked before acceptance.
5. [_validated_ontology_projection](../../agents/ra/composition_context.py) — line 387: completed assertions, selected assignment and fingerprints enter; a matching accepted ontology projection is returned to RA context.

## Read this test

[test_project_tbox_is_the_exact_schema_only_spec2primitives_vocabulary](../../tests/test_ppr_tbox.py) — line 27: asserts the 12 classes, 12 object properties, zero declared datatype properties, exact property directions, four functional properties and pairwise cardinality.

[test_current_proposals_attach_pairwise_relationships_to_their_stated_state](../../tests/test_pa_grounding_contracts.py) — line 511: covers 0, 1 and 3 associations, each with two endpoints and the requested state linkage. [test_endpoints_can_share_a_state_without_creating_an_allocation_pair](../../tests/test_pa_grounding_contracts.py) — line 640: demonstrates that endpoint binding does not dictate current-versus-desired relationship membership. These tests were read, not rerun.

## Runtime evidence

The [saved ABox](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology/interaction_abox.ttl), [proposal](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json) and [completion](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/context_completion_0001.json) belong to the original `assemble medium gear.` interaction. The ABox, proposal, assignment delta and Workcell snapshot hashes matched their completion pins during source checks. Exact names, owner types and ontology terms were preserved. No ground-truth evaluator answers were read.

## You can ignore

- Shared `SystemBridge`, ProductAgent and RobotAgent internals.
- Phase 5 target calculation, primitive refinement and execution implementation.
- Ground-truth evaluation data and unrelated saved runs.

## Refactoring performed

None in the repository. The previous temporary presentation renderer was reused to create editable diagrams and matching PDF layouts.

## Verification and intentionally unchanged behavior

Presentation checks: parsed the supplied OWL and cited Turtle graph; verified the cited record hashes and exact property profiles; reopened the PowerPoint with 15 slides and presenter notes; verified 9 visible main slides and 6 hidden backup slides; checked 15 PDF pages, 600-second main timing, links and text layout; visually inspected renders; ran `git diff --check`. The PDF uses the same layout data as the PowerPoint. Microsoft PowerPoint/LibreOffice rendering was unavailable; the PowerPoint package was structurally reopened and checked.

No Python runtime tests, full compile checks, live model calls, ROS2/MoveIt runs or hardware tests were run for this presentation-only task. Original overview slides, saved interactions, schema files and runtime behavior remain unchanged. Existing local edits in PA/RA, adapters, UI, tests and geometry were preserved separately from these three new presentation files.
