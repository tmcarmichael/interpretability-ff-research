# Results directory

Every JSON file here is a committed experimental result. Analysis scripts read from these files via `analysis/load_results.py`.

## Provenance and reproducibility

Every result file carries a complete provenance record: the model identifier and Hugging Face revision commit, the producing script, the execution environment, and the protocol parameters (`eval_seeds`, `target_ex_per_dim`, `batch_size`) used to generate the file. The `provenance.value_source` field declares how the provenance values were recorded: `"runtime"` (captured by the producing script at experiment time) or `"post_hoc_deterministic"` (set deterministically from version-controlled code and commit history). Measurement values (`partial_corr`, `output_controlled`, `control_sensitivity`, `baselines`, `layer_profile`, `ablation_results`, and every per-seed array) are original.

Every reported number is reproducible by running the recorded script with the recorded Hugging Face revision on a CUDA device. Software baseline: Python 3.12, PyTorch 2.8, CUDA 12.8.

## Cross-family scaling

| File | Model | Protocol |
|---|---|---|
| `transformer_observe.json` | GPT-2 124M-1.5B | 3-20 seeds, full battery; for GPT-2 XL the v3 numbers come from the `gpt2-xl_main.json` overlay below |
| `gpt2-xl_main.json` | GPT-2 XL (1.5B) | 7-seed v3 overlay; consumed by `_load_gpt2` in `analysis/load_results.py` |
| `qwen2.5-0.5b_main.json` | Qwen 2.5 0.5B | 7-seed, 600 ex/dim |
| `qwen2.5-0.5b-instruct_main.json` | Qwen 2.5 0.5B Instruct | 7-seed, 600 ex/dim |
| `qwen2.5-1.5b_main.json` | Qwen 2.5 1.5B | 7-seed, 350 ex/dim |
| `qwen2.5-1.5b-instruct_main.json` | Qwen 2.5 1.5B Instruct | 7-seed, 350 ex/dim |
| `qwen2.5-3b_main.json` | Qwen 2.5 3B | 7-seed, 350 ex/dim |
| `qwen2.5-3b-instruct_main.json` | Qwen 2.5 3B Instruct | 7-seed, 350 ex/dim |
| `qwen2.5-7b_main.json` | Qwen 2.5 7B | 7-seed, 350 ex/dim |
| `qwen2.5-7b-instruct_main.json` | Qwen 2.5 7B Instruct | 7-seed, 350 ex/dim |
| `qwen2.5-14b_main.json` | Qwen 2.5 14B | 7-seed, 350 ex/dim |
| `qwen2.5-14b-instruct_main.json` | Qwen 2.5 14B Instruct | 7-seed, 350 ex/dim |
| `qwen2.5-32b_main.json` | Qwen 2.5 32B | 7-seed, 350 ex/dim |
| `gemma-3-1b_main.json` | Gemma 3 1B | 7-seed, 150 ex/dim |
| `gemma-3-4b_main.json` | Gemma 3 4B | 7-seed, 350 ex/dim |
| `llama-3.2-1b_main.json` | Llama 3.2 1B | 7-seed, 350 ex/dim |
| `llama-3.2-1b-instruct_main.json` | Llama 3.2 1B Instruct | 7-seed, 350 ex/dim |
| `llama-3.2-3b_main.json` | Llama 3.2 3B | 7-seed, 350 ex/dim |
| `llama-3.1-8b_main.json` | Llama 3.1 8B | 7-seed, 350 ex/dim |
| `mistral-7b-v0.3_main.json` | Mistral 7B v0.3 | 7-seed, 350 ex/dim |
| `mistral-7b-instruct-v0.3_main.json` | Mistral 7B Instruct v0.3 | 7-seed, 350 ex/dim |
| `phi-3-mini_main.json` | Phi-3 Mini 4K Instruct | 7-seed, 350 ex/dim |

## Pythia suite (within-recipe controlled)

| File | Model | Notes |
|---|---|---|
| `pythia-70m_main.json` | Pythia 70M | Includes random-probe baseline |
| `pythia-160m_main.json` | Pythia 160M | 7-seed |
| `pythia-410m_main.json` | Pythia 410M | Collapse configuration (24L, 16H) |
| `pythia-1b_main.json` | Pythia 1B | 7-seed |
| `pythia-1.4b_main.json` | Pythia 1.4B | Collapse configuration (24L, 16H) |
| `pythia-1.4b-deduped_main.json` | Pythia 1.4B deduped Pile | Collapse replication across corpora |
| `pythia-1.4b_shuffle-control.json` | Pythia 1.4B | Shuffled-label null distribution |
| `pythia-2.8b_main.json` | Pythia 2.8B | 7-seed |
| `pythia-6.9b_main.json` | Pythia 6.9B | 7-seed |
| `pythia-12b_main.json` | Pythia 12B | 7-seed |

## Checkpoint dynamics (within-recipe controlled)

| File | Model | Notes |
|---|---|---|
| `pythia-1b_dynamics.json` | Pythia 1B (16L/8H, d=2048) | 10 checkpoints, step 256 to 143000; healthy trajectory |
| `pythia-1.4b_dynamics.json` | Pythia 1.4B (24L/16H, d=2048) | 10 checkpoints, step 256 to 143000; collapse trajectory |

Matched hidden dimension ($d = 2048$). Both start healthy at step 256; the 1B recovers after a mid-training dip, the 1.4B converges collapsed. Per-checkpoint fields: partial_corr (7-seed), output_controlled (3-seed), perplexity, peak layer, and Hugging Face revision hash. These files use a different schema from the single-model results and are validated separately by `validate_dynamics_json` in `analysis/load_results.py`.

## Pythia training corpus

Pythia checkpoints are pre-trained models loaded from Hugging Face with revisions pinned in `model_revisions.json`; this paper does not retrain Pythia on any corpus. Reproduction requires the pre-trained checkpoints and the evaluation datasets pinned in `dataset_revisions.json`, not the original training corpus. Two corpus variants are represented in the suite: the standard Pile (`pythia-1.4b_main.json`) and the deduplicated Pile (`pythia-1.4b-deduped_main.json`, the variant trained from `EleutherAI/the_pile_deduplicated`). The 1.4B and 1.4B-deduped collapse values differ by less than the inter-seed range, so corpus deduplication is not the load-bearing variable for the collapse phenomenon.

## Downstream tasks

| File | Task | Model |
|---|---|---|
| `qwen2.5-7b-instruct_squad-rag.json` | SQuAD 2.0 RAG | Qwen 7B Instruct |
| `phi-3-mini_squad-rag.json` | SQuAD 2.0 RAG | Phi-3 Mini Instruct |
| `mistral-7b-instruct-v0.3_squad-rag.json` | SQuAD 2.0 RAG | Mistral 7B Instruct |
| `qwen2.5-7b-instruct_medqa.json` | MedQA-USMLE | Qwen 7B Instruct |
| `phi-3-mini_medqa.json` | MedQA-USMLE | Phi-3 Mini Instruct |
| `mistral-7b-instruct-v0.3_medqa.json` | MedQA-USMLE | Mistral 7B Instruct |
| `qwen2.5-7b-instruct_truthfulqa.json` | TruthfulQA | Qwen 7B Instruct |
| `phi-3-mini_truthfulqa.json` | TruthfulQA | Phi-3 Mini Instruct |
| `mistral-7b-instruct-v0.3_truthfulqa.json` | TruthfulQA | Mistral 7B Instruct |

## Probe-validity controls

| File | Model | Test |
|---|---|---|
| `gpt-2-124m_shuffle-control.json` | GPT-2 124M | 10 permutations, shuffled labels |
| `qwen2.5-7b_width-sweep.json` | Qwen 2.5 7B | Output predictor 64-512 units |
| `qwen2.5-0.5b_exdim-sweep.json` | Qwen 2.5 0.5B | Token budget sensitivity 150-1000 ex/dim |
| `qwen2.5-7b_bootstrap.json` | Qwen 2.5 7B | 30-resample document-level bootstrap |
| `gpt-2-124m_nonlinear-probe.json` | GPT-2 124M | Linear vs MLP |
| `qwen2.5-0.5b_nonlinear-probe.json` | Qwen 2.5 0.5B | Linear vs MLP |
| `qwen2.5-1.5b_nonlinear-probe.json` | Qwen 2.5 1.5B | Linear vs MLP |
| `qwen2.5-3b_nonlinear-probe.json` | Qwen 2.5 3B | Linear vs MLP |
| `qwen2.5-7b_nonlinear-probe.json` | Qwen 2.5 7B | Linear vs MLP |
| `qwen2.5-14b_nonlinear-probe.json` | Qwen 2.5 14B | Linear vs MLP |
| `gemma-3-1b_nonlinear-probe.json` | Gemma 3 1B | Linear vs MLP |
| `llama-3.2-3b_nonlinear-probe.json` | Llama 3.2 3B | Linear vs MLP, held-out HP selection |
| `llama-3.2-3b_nonlinear-probe-multilayer.json` | Llama 3.2 3B | 5-layer sweep |
| `pythia-410m_nonlinear-probe.json` | Pythia 410M | Collapse-point MLP comparison |
| `pythia-1.4b_nonlinear-probe.json` | Pythia 1.4B | Collapse-point MLP comparison |

## Mechanistic analysis

| File | Model | Analysis |
|---|---|---|
| `qwen2.5-7b_mechanistic.json` | Qwen 2.5 7B | Mean-ablation patching |
| `llama-3.2-1b_mechanistic.json` | Llama 3.2 1B | Mean-ablation patching |
| `llama-3.2-3b_mechanistic.json` | Llama 3.2 3B | Mean-ablation patching |
| `mistral-7b-v0.3_mechanistic.json` | Mistral 7B v0.3 | Mean-ablation patching |

## Training-time interventions

| File | Model | Notes |
|---|---|---|
| `qwen2.5-0.5b_auxiliary-loss.json` | Qwen 2.5 0.5B | Naive auxiliary-loss fine-tune (lambda=0.05, 300 steps, lr=2e-5); cited in discussion.tex as evidence that observability-preserving training is non-trivial |

## Comparison baselines and library-only data

| File | Model | Role |
|---|---|---|
| `qwen2.5-0.5b_exdim-1000.json` | Qwen 2.5 0.5B | Comparison baseline at 1000 ex/dim, cited in appendix B token-budget discussion |

## Paper-cited predecessor data

Predecessor MLP/FF work whose results are still cited by paper macros.

| File | Cited via |
|---|---|
| `gpt-2_bottleneck-scaling.json` | `\bottleneck*` macros (output-controlled proportional sweep) |
| `gpt-2-124m_sae-comparison.json` | `\sae*` macros (SAE vs raw probe, 3 seeds) |

## Infrastructure

| File | Purpose |
|---|---|
| `model_revisions.json` | Hugging Face commit hashes for all models |
| `dataset_revisions.json` | Hugging Face dataset config IDs and commit hashes for every paper-cited evaluation corpus (WikiText-103, C4, SQuAD 2.0, MedQA-USMLE, TruthfulQA) |

## JSON schema

Every full-protocol result contains: `model`, `n_layers`, `hidden_dim`, `protocol`, `peak_layer_final`, `layer_profile`, `partial_corr` (with `per_seed`), `test_split_comparison`, `seed_agreement`, `output_controlled`, `baselines`, `cross_domain`, `control_sensitivity`, `flagging_6a`. See `analysis/load_results.py` for the schema validator.

## Macro provenance

Every paper-cited value in the published PDF resolves through a `\newcommand` in `data_macros.sty` to either a single cell in one of the JSONs in this directory or to an aggregation produced by the analysis library. The trace lives in two places:

- `data_macros.sty` (shipped in the arXiv source bundle) carries a trailing comment on every `\newcommand` line naming the source experiment.
- Each result JSON's `provenance.script` field names the script that produced its cells; the analysis library under `analysis/` produces aggregations across cells.

To verify a paper-cited value:

1. Locate the macro in `data_macros.sty`. The trailing comment names the source experiment.
2. If the comment names a single experiment, open the corresponding result JSON in this directory and follow the key path implied by the macro role. Example: `\corepcorr` resolves to `transformer_observe.json` at `8.models.gpt2.partial_corr.mean`.
3. If the macro is computed from multiple cells (typical for means, ratios, and percentages spanning families), the aggregating logic is in `analysis/load_results.py` and the modules in `analysis/`. To regenerate every aggregate, run `uv sync && uv run python analysis/run_all.py`.
