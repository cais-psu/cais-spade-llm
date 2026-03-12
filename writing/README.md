# Writing

This directory organizes all paper-writing artifacts for the CAIS-SPADE-LLM project.

## Structure

```
writing/
├── outline.md               # Paper outline mapped to code
├── README.md                 # This file
├── drafts/                   # Per-section working drafts
│   └── README.md
├── experiments/              # Experiment scaffolding
│   ├── README.md
│   ├── scenarios/            # Failure injection configs
│   ├── results/              # Raw experiment outputs
│   ├── analysis/             # Scripts: results → figures
│   └── figures/              # Publication-ready plots
└── figures/                  # Architecture/concept diagrams
    └── README.md
```

## Workflow

1. **Start with** [`outline.md`](outline.md) — it maps every paper section to the code that supports it
2. **Design experiments** using [`experiments/README.md`](experiments/README.md) and the detailed plans in [`TODO/experimental_setup_and_results.md`](../TODO/experimental_setup_and_results.md)
3. **Write section drafts** in [`drafts/`](drafts/) — one file per section
4. **Run experiments** → collect in `experiments/results/` → process with `experiments/analysis/` → output to `experiments/figures/`
5. **Assemble** into final manuscript

## Relationship to TODO/

The `TODO/` folder contains technical design docs that double as proto-paper-sections:

| TODO File | Maps To |
|-----------|---------|
| `experimental_setup_and_results.md` | §6 Experiments |
| `safety_constraints_in_replanning.md` | §5 Recovery (DFA in BFS) + §4 Safety |
| `shared_typed_ir_for_product_and_safety.md` | §7 Discussion (future work) |
| `des_bidding_for_offline_planning.md` | §8 Future Work |
| `validate_llm_bridge_output.md` | §5.2 LLM Bridge Validation |
