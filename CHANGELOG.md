# Changelog

All notable changes to this repository.

## v4.0.1 (2026-05-05)

Untrack `notebooks/`. No code or results changed. v4.0.0 remains the arXiv v2 artifact.

## v4.0.0 (2026-05-05) — arXiv v2

Companion code release for arXiv v2 of the paper, retitled
**"Architectural Observability Collapse in Transformers"** (was
"Architecture Determines Observability in Transformers" in arXiv v1).

### New experiments

- **Residualizer-fit split robustness** (`scripts/run_residualizer_split.py`).
  Tests whether the OLS residualizer overfits to the probe-training pool
  by fitting on a disjoint document pool R and applying without refit
  to the probe-training and evaluation pools. Run on four
  representative regimes: GPT-2 124M (healthy anchor), Pythia 1B
  (healthy controlled), Pythia 1.4B (controlled collapse), Llama 3.2
  3B (observational collapse). All four satisfy the pre-specified
  regime-preservation criterion; max same-pool $|\Delta\rho_{\rm partial}| = 0.017$.
  Results in `results/*_residualizer-split.json`.

- **Pythia 1B and 1.4B checkpoint dynamics** at matched hidden
  dimension (d=2048) across 10 checkpoints from 0.5B to 300B tokens.
  Both configurations form the signal at the earliest measured
  checkpoint; training erases it in 1.4B (24L/16H) while 1B (16L/8H)
  recovers. Results in `results/pythia-1b_dynamics.json` and
  `results/pythia-1.4b_dynamics.json`.

- **Gemma 3 1B canonical-protocol rerun** at 350 ex/dim. The earlier
  150 ex/dim measurement was under-saturated; the canonical-protocol
  rerun produces $\rho_{\rm partial} = 0.216$ with normal random-baseline
  behavior and a mid-layer peak at L11. Replaces previous results.

- **Qwen 2.5 32B** added to the cross-family cohort. Within-Qwen
  observability now characterized across a 64x parameter range
  (0.5B through 32B, six base sizes). Result in
  `results/qwen2.5-32b_main.json`.

- **GPT-2 family canonical-protocol reruns** (124M, 355M, 774M, 1.5B)
  at 350 ex/dim, 7-seed protocol with output-controlled residual.
  Replaces the earlier non-uniform per-size protocol used in v3.3.x.
  Results in `results/gpt2-{124m,medium,large,xl}_main.json`.

- **Shuffle-control replication** (`results/gpt-2-124m_shuffle-control.json`).
  10-permutation shuffle of binary targets on GPT-2 124M; backs the
  five-sigma shuffle null cited in the validity argument.

### Reproducibility infrastructure

- **Cross-repo claim provenance** in `reports/`:
  - `reports/paper_values.json`: every paper-cited macro with its
    value, description, section, source files, key paths, formula,
    and named scope. 271 macros total; 118 with full source-file +
    key-path + formula provenance, including all paper headline
    values. Coverage floor enforced by tests; cannot regress without
    explicit consent.
  - `reports/scopes.json`: named cohort definitions
    (`cross_family_14`, `pythia_controlled_9`) mirrored from
    `analysis/load_results.py:SCOPES`, with drift test.
  - `reports/figure_sources.json`: every committed PDF figure mapped
    to its generator script and source JSONs.

- **Formal JSON Schemas** in `schema/` (Draft 2020-12) for every
  result type: main, dynamics, residualizer-split, nonlinear-probe,
  mechanistic, downstream, shuffle-control, bootstrap, width-sweep,
  and legacy. Dispatched by filename pattern in
  `scripts/validate_schemas.py:DISPATCH`. CI validates every
  committed result JSON against its dispatched schema.

- **Manifest verification**: every entry in
  `results/model_revisions.json` is programmatically verified against
  the Hugging Face API. Latest report at
  `results/manifest_verification/2026-05-03.json`. Regeneratable via
  `scripts/verify_manifest_revisions.py`.

- **Croissant 1.1 manifest** at `croissant.json` exposes the
  result-file dataset to ML benchmark indexers
  ([mlcommons.org/croissant](https://mlcommons.org/croissant)).
  Validated against the `mlcroissant` reference spec on every CI run
  via `just check-croissant`.

- **CUDA enforcement** for paper-cited result JSONs. The schema
  enforces `provenance.device == "cuda"` on every committed
  `*_main.json`, `*_dynamics.json`, and `*_residualizer-split.json`.
  MPS is allowed for local development only.

- **Dataset revision pinning** via `results/dataset_revisions.json`.
  Every `load_dataset(...)` call in producer scripts passes a pinned
  revision; `tests/test_script_preflight.py` enforces this repo-wide.

- **Test suite expansion** in `tests/test_paper_values.py` (97 tests,
  2 module-level skips for missing artifacts):
  - Schema validation against `schema/main.schema.json` and
    `schema/dynamics.schema.json`
  - Provenance integrity (every `source_files` entry resolves; every
    `key_paths` walks; every named scope exists)
  - Direct-read auto-verification (every macro tagged
    `formula="direct read"` matches its JSON cell at the formatted
    precision)
  - Idempotency of all three exporters
  - Scope membership matches `analysis/load_results.py:SCOPES`
  - Figure source-file existence
  - `paper_version` consistency between
    `reports/paper_values.json` and the paper repo's
    `main.tex` `\paperversion`

### Methodology

- **Canonical 350 ex/dim protocol** standardized across all
  paper-cited result JSONs. Cross-family scope is 14 models in 6
  families.

- **20-seed statistical hardening** on GPT-2 124M (layer 11):
  $\rho_{\rm partial} = 0.282 \pm 0.001$, 95% CI [0.282, 0.283],
  per-seed range [0.279, 0.284], seed agreement 0.993.

- **Three-pool protocol** (R/T/V) for the residualizer-fit split:
  disjoint WikiText-103 documents at the same hidden-dim token
  budget as the canonical protocol.

### Removed

- **Mechanistic ablation appendix (Appendix G)** from the paper:
  directional ablation, mean-ablation patching, and layer formation
  of the output-independent component were exploratory and not
  load-bearing for any Paper 1 claim. Mechanism work is the scope of
  Paper 2 (`nn-mechanistic`). The corresponding result JSONs
  (`*_mechanistic.json`) and `transformer_observe.json` mechanistic
  blocks remain in this repository as historical artifacts; they are
  no longer cited by the paper.

- **Layer-1 ablation pattern subsection (A.12)** from the
  Methodology hardening appendix: the Llama 3.2 1B vs 3B and Mistral
  7B sign-reversal observation was exploratory and removed for the
  same reason as Appendix G.

### Documentation

- **Reviewer-facing verification path** in `README.md`: three
  independent paths (structured claim provenance, targeted CLI
  verification, full pipeline), with worked example walking
  `confabsorbmean` from `paper_values.json` through the 14 source
  JSONs to recompute.

### Numerical updates

- Confidence absorption recomputed under canonical protocol: $60.3\%
  \pm 7.0\%$ across 14 models in 6 families (was 58.2% in v3.3.x;
  drift due to the cohort change).

### v3.3.0 (arXiv v1) baseline

The arXiv v1 release. Tagged `arxiv-v1`. Every change above is
relative to that baseline.

---

For full provenance of every paper-cited number, see
`reports/paper_values.json` (regenerated with each release). For
schema-validated reproducibility of every result file, see `schema/`.
