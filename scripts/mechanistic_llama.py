"""Mechanistic analysis: Llama 3.2 1B vs 3B via mean-ablation patching.

Compares the high-observability 1B model against the low-observability 3B
model within the same family. Tests whether the 3B signal is never
generated or generated and then suppressed by a specific component.
"""

import datetime as _dt
import gc
import json
import sys
import time
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Add src/ to path for transformer_observe import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _resolve_out(name_or_path):
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    base = (
        Path("/workspace")
        if Path("/workspace").exists()
        else Path(__file__).resolve().parent.parent / "results"
    )
    return base / p


def _revision_kwargs(model_id):
    manifest = Path(__file__).resolve().parent.parent / "results" / "model_revisions.json"
    if not manifest.exists():
        return {}
    commit = json.loads(manifest.read_text()).get("models", {}).get(model_id, {}).get("commit")
    return {"revision": commit} if commit else {}


TARGET_PATHS = {
    "llama_1b": _resolve_out("llama-3.2-1b_mechanistic.json"),
    "llama_3b": _resolve_out("llama-3.2-3b_mechanistic.json"),
}
for _p in TARGET_PATHS.values():
    _p.parent.mkdir(parents=True, exist_ok=True)
for _label, _p in TARGET_PATHS.items():
    print(f"Output {_label}: {_p}")

from transformer_observe import (
    load_wikitext,
    run_mechanistic_general,
)

if not torch.cuda.is_available():
    import sys

    sys.exit(
        "mechanistic_llama.py produces paper-quality results and requires CUDA. "
        "Run on a CUDA-enabled host (Colab GPU, runpod, local CUDA box)."
    )
DEVICE = "cuda"
RUN_START = time.time()


def elapsed():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


MODELS = [
    {
        "model_id": "meta-llama/Llama-3.2-1B",
        "peak_layer": 13,
        "label": "llama_1b",
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B",
        "peak_layer": 14,
        "label": "llama_3b",
    },
]

# Fail-fast before model download.
if not RESULTS_DIR.is_dir():
    sys.exit(f"RESULTS_DIR not found: {RESULTS_DIR}")
manifest = RESULTS_DIR / "model_revisions.json"
if not manifest.is_file():
    sys.exit(f"Manifest missing: {manifest}")
_known = json.loads(manifest.read_text()).get("models", {})
for _spec in MODELS:
    if _spec["model_id"] not in _known:
        sys.exit(f"Model {_spec['model_id']!r} not in manifest")

print(f"Device: {DEVICE}")
print(f"Models: {[m['label'] for m in MODELS]}")

# Load WikiText once
print(f"\nLoading WikiText-103... [{elapsed()}]")
train_docs = load_wikitext("train", max_docs=2000)
test_docs = load_wikitext("test", max_docs=500)
print(f"  {len(train_docs)} train, {len(test_docs)} test docs")

results = {}

for spec in MODELS:
    model_id = spec["model_id"]
    peak_layer = spec["peak_layer"]
    label = spec["label"]

    print(f"\n{'=' * 60}")
    print(f"  Loading {model_id} [{elapsed()}]")
    print(f"{'=' * 60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _rev_kw = _revision_kwargs(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **_rev_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **_rev_kw).to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.1f}B params, {n_layers} layers, {hidden_dim} dim")

    # Scale token budgets by hidden dim
    min_ex_per_dim = 150
    adj_train = max(min_ex_per_dim * hidden_dim, 200000)
    adj_test = max(min_ex_per_dim * hidden_dim // 2, 100000)

    print(f"  Token budget: train={adj_train}, eval_budget=15000")
    print(f"  Peak layer: {peak_layer} ({peak_layer / n_layers * 100:.0f}% depth)")

    mech = run_mechanistic_general(
        model,
        tokenizer,
        DEVICE,
        train_docs,
        test_docs,
        adj_train,
        adj_test,
        peak_layer=peak_layer,
        eval_budget=15000,
    )

    mech["model"] = model_id
    mech["label"] = label
    mech["n_params_b"] = round(n_params / 1e9, 1)
    mech["n_layers"] = n_layers
    mech["hidden_dim"] = hidden_dim
    _revision = _rev_kw.get("revision") or getattr(model.config, "_commit_hash", None)
    if not _revision:
        raise RuntimeError(
            f"Could not resolve model revision for {model_id}: pin via results/model_revisions.json "
            "or upgrade transformers (model.config._commit_hash unset)."
        )
    mech["provenance"] = {
        "model_revision": _revision,
        "script": "scripts/mechanistic_llama.py",
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "value_source": "runtime",
        "device": str(DEVICE),
    }
    results[label] = mech

    # Free memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  {label} complete [{elapsed()}]")

# Save per-model files matching the schema of other mechanistic results
# (qwen2.5-7b_mechanistic.json, mistral-7b-v0.3_mechanistic.json).
for label, mech in results.items():
    target = TARGET_PATHS[label]
    with open(target, "w") as f:
        json.dump(mech, f, indent=2)
    print(f"\nSaved {target}")
print(f"Total time: {elapsed()}")

# Quick comparison summary
print(f"\n{'=' * 60}")
print("  COMPARISON SUMMARY")
print(f"{'=' * 60}")
for label, mech in results.items():
    print(f"\n  {label} (peak L{mech['peak_layer']}):")
    ablation = mech["ablation_results"]
    # Find top 3 components by absolute obs_resid_delta
    all_components = []
    for layer_str, comps in ablation.items():
        for comp_name, vals in comps.items():
            all_components.append((int(layer_str), comp_name, vals["obs_resid_delta"]))
    all_components.sort(key=lambda x: abs(x[2]), reverse=True)
    print("  Top 3 components by |obs_resid_delta|:")
    for layer, comp, delta in all_components[:3]:
        print(f"    L{layer} {comp}: {delta:+.4f}")

    if mech.get("composition_results"):
        print("  Composition (all_top):")
        at = mech["composition_results"].get("all_top", {})
        if at:
            print(
                f"    actual={at['obs_resid_delta']:+.4f}  expected={at['expected_additive']:+.4f}  interaction={at['interaction']:+.4f}"
            )
