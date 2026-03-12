# Experiments

This directory holds everything needed to run paper experiments and produce publication-ready results.

## Structure

```
experiments/
├── scenarios/       # Input configs for each failure injection test
├── results/         # Raw output logs, CSVs, JSON from experiment runs
├── analysis/        # Scripts that process results → tables and plots
└── figures/         # Output plots/charts ready for the paper
```

## Experiment Index

Full experiment design lives in [`../../TODO/experimental_setup_and_results.md`](../../TODO/experimental_setup_and_results.md).

| # | Name | Priority | Status |
|---|------|----------|--------|
| 1 | Layer Effectiveness (DES vs LLM vs Full) | **Critical** | Not started |
| 2 | Validation Pipeline Ablation | High | Not started |
| 3 | Recovery Latency Breakdown | High | Not started |
| 4 | Safety Constraint Enforcement | **Critical** | Not started |
| 5 | Scalability | Medium | Not started |
| 6 | DES vs LLM for Offline Planning | Low (future work) | Not started |

## Running Experiments

> **TODO**: Add automation scripts here as experiments are implemented.

The general flow is:

1. Define scenario configs in `scenarios/`
2. Run the system with failure injection enabled
3. Collect logs into `results/`
4. Run analysis scripts from `analysis/` to produce figures
