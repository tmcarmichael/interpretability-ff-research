"""Residualizer-fit split protocol.

Tests whether the OLS residualizer used to construct binary targets is
overfitting to the probe-training pool. The standard protocol fits OLS
coefficients on the same documents used to train the probe (T). Here we
fit on a held-out third pool R, then apply those coefficients to T
without re-centering. R, T, V draw from disjoint document populations.

Single self-contained script; can be dropped into RunPod /workspace and
invoked as `python run_residualizer_split.py ...`.

Output JSON contains both Main rho_partial (beta fit on T) and Split
rho_partial (beta fit on R, applied to T), pos-label fractions for
both, and per-seed values.
"""

import argparse
import datetime as _dt
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_wikitext(split, max_docs=None, dataset_revisions=None):
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        revision=dataset_revisions["Salesforce/wikitext"]["commit"],
        streaming=bool(max_docs),
    )
    docs, current = [], []
    for row in ds:
        text = row["text"]
        if text.strip() == "" and current:
            docs.append("\n".join(current))
            current = []
            if max_docs and len(docs) >= max_docs:
                break
        elif text.strip():
            current.append(text)
    if current:
        docs.append("\n".join(current))
    return docs


def pretokenize(docs, tokenizer, max_length=512):
    encoded = []
    for doc in docs:
        if not doc.strip():
            continue
        ids = tokenizer.encode(doc, truncation=True, max_length=max_length)
        if len(ids) >= 2:
            encoded.append(ids)
    encoded.sort(key=len)
    return encoded


def build_batches(encoded, batch_size):
    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        max_len = len(chunk[-1])
        B = len(chunk)
        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long)
        for j, ids in enumerate(chunk):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn_mask[j, : len(ids)] = 1
        batches.append((input_ids, attn_mask))
    return batches


def partial_spearman(x, y, covariates):
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, p = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r), float(p)


def fit_residualizer(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    return np.linalg.lstsq(X, losses, rcond=None)[0]


def apply_residualizer(losses, max_softmax, activation_norm, beta):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    return losses - X @ beta


def _get_layer_list(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def collect_at_layer(model, batches, layer, max_tokens, device, want_activations, sm_chunk=8):
    """Collect losses, max_softmax, activation_norm at one layer.

    When want_activations is True, also collects the activation tensor.
    """
    model.eval()
    layer_modules = _get_layer_list(model)
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured[layer] = h

    handle = layer_modules[layer].register_forward_hook(hook_fn)

    acts_buf, norms_buf = [], []
    losses_buf, sm_buf = [], []
    total = 0

    for bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
        if total >= max_tokens:
            break
        input_ids = input_ids_cpu.to(device)
        attn_mask = attn_mask_cpu.to(device)
        B, S = input_ids.shape
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
        shift_mask = attn_mask[:, 1:].bool()
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        V = shift_logits.size(-1)
        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            B, S - 1
        )
        sm_2d = torch.empty(B, S - 1, device=device)
        for ci in range(0, B, sm_chunk):
            p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
            sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
            del p
        losses_buf.append(losses_2d[shift_mask].float().cpu())
        sm_buf.append(sm_2d[shift_mask].float().cpu())
        h = captured[layer][:, :-1, :].float()
        norms_buf.append(h.norm(dim=-1)[shift_mask].cpu())
        if want_activations:
            acts_buf.append(h[shift_mask].cpu())
        total += shift_mask.sum().item()
        captured.pop(layer, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels
        del losses_2d, sm_2d, shift_mask, h
        if device == "cuda":
            torch.cuda.empty_cache()
        if (bi + 1) % 10 == 0:
            print(f"      batch {bi + 1}/{len(batches)}, {total} positions")

    handle.remove()
    n = min(total, max_tokens)
    out = {
        "losses": torch.cat(losses_buf).numpy()[:n],
        "max_softmax": torch.cat(sm_buf).numpy()[:n],
        "activation_norm": torch.cat(norms_buf).numpy()[:n],
    }
    if want_activations:
        out["activations"] = torch.cat(acts_buf)[:n]
    return out


def train_probe(train_data, beta, seed, train_device, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].to(train_device)
    residuals = apply_residualizer(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"], beta
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(train_device)
    pos_frac = float(targets.mean().item())
    head = torch.nn.Linear(acts.size(1), 1).to(train_device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu(), pos_frac


def evaluate_head(head, test_data):
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, _ = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return rho


parser = argparse.ArgumentParser(description="Residualizer-fit split protocol.")
parser.add_argument("--model", required=True, help="HF model ID")
parser.add_argument("--output", required=True, help="Output JSON name")
parser.add_argument("--layer", type=int, required=True, help="FINAL layer from main protocol")
parser.add_argument("--ex-dim", type=int, default=350)
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--seeds", type=int, default=3, help="Number of probe seeds (starts at 43)")
parser.add_argument(
    "--max-train-docs",
    type=int,
    default=24000,
    help="Total train docs loaded; first half = T, second half = R",
)
parser.add_argument("--trust-remote-code", action="store_true")
parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
args = parser.parse_args()

if not torch.cuda.is_available():
    sys.exit("CUDA required.")
DEVICE = "cuda"
TRAIN_DEVICE = "cuda"

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

# Fail-fast before model download.
if not RESULTS_DIR.is_dir():
    sys.exit(f"RESULTS_DIR not found: {RESULTS_DIR}")
manifest = RESULTS_DIR / "model_revisions.json"
if not manifest.is_file():
    sys.exit(f"Manifest missing: {manifest}")
if args.model not in json.loads(manifest.read_text()).get("models", {}):
    sys.exit(f"Model {args.model!r} not in manifest")
ds_manifest = RESULTS_DIR / "dataset_revisions.json"
if not ds_manifest.is_file():
    sys.exit(f"Dataset manifest missing: {ds_manifest}")
DATASET_REVISIONS = json.loads(ds_manifest.read_text()).get("datasets", {})
model_revision = json.loads(manifest.read_text()).get("models", {}).get(args.model, {}).get("commit")
_rev_kw = {"revision": model_revision} if model_revision else {}
dataset_revisions = DATASET_REVISIONS


def _resolve_out(name_or_path):
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    base = Path("/workspace") if Path("/workspace").exists() else RESULTS_DIR
    return base / p


OUT_PATH = _resolve_out(args.output)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Output: {OUT_PATH}")
print(f"Model revision: {model_revision or '(unpinned)'}")

RUN_START = time.time()


def elapsed():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


from transformers import AutoModelForCausalLM, AutoTokenizer

load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": args.attn_impl, **_rev_kw}
if args.trust_remote_code:
    load_kwargs["trust_remote_code"] = True

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, **_rev_kw)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(DEVICE)
model.eval()
if not model_revision:
    model_revision = getattr(model.config, "_commit_hash", None)

_cfg = getattr(model.config, "text_config", model.config)
N_LAYERS = _cfg.num_hidden_layers
HIDDEN_DIM = _cfg.hidden_size
MAX_TOKENS = args.ex_dim * HIDDEN_DIM
print(f"{N_LAYERS} layers, {HIDDEN_DIM} dim, target_layer=L{args.layer}, max_tokens={MAX_TOKENS}")

print(f"\n=== Loading wikitext [{elapsed()}] ===")
train_docs = load_wikitext("train", max_docs=args.max_train_docs, dataset_revisions=dataset_revisions)
val_docs = load_wikitext("validation", dataset_revisions=dataset_revisions)
print(f"  {len(train_docs)} train docs, {len(val_docs)} val docs")
half = len(train_docs) // 2
T_docs = train_docs[:half]
R_docs = train_docs[half:]
print(f"  T={len(T_docs)}, R={len(R_docs)}")

T_enc = pretokenize(T_docs, tokenizer)
R_enc = pretokenize(R_docs, tokenizer)
V_enc = pretokenize(val_docs, tokenizer)
T_batches = build_batches(T_enc, args.batch_size)
R_batches = build_batches(R_enc, args.batch_size)
V_batches = build_batches(V_enc, args.batch_size)
print(f"  T={len(T_batches)} batches, R={len(R_batches)}, V={len(V_batches)}")

print(f"\n=== Collecting T at L{args.layer} [{elapsed()}] ===")
T_data = collect_at_layer(model, T_batches, args.layer, MAX_TOKENS, DEVICE, want_activations=True)
print(f"\n=== Collecting V at L{args.layer} [{elapsed()}] ===")
V_data = collect_at_layer(model, V_batches, args.layer, MAX_TOKENS, DEVICE, want_activations=True)
print(f"\n=== Collecting R at L{args.layer} (no activations) [{elapsed()}] ===")
R_data = collect_at_layer(model, R_batches, args.layer, MAX_TOKENS, DEVICE, want_activations=False)

del model, T_batches, R_batches, V_batches
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

beta_T = fit_residualizer(T_data["losses"], T_data["max_softmax"], T_data["activation_norm"])
beta_R = fit_residualizer(R_data["losses"], R_data["max_softmax"], R_data["activation_norm"])
print(f"\nbeta_T (sm, norm, intercept) = {beta_T}")
print(f"beta_R (sm, norm, intercept) = {beta_R}")

res_main = apply_residualizer(T_data["losses"], T_data["max_softmax"], T_data["activation_norm"], beta_T)
res_split = apply_residualizer(T_data["losses"], T_data["max_softmax"], T_data["activation_norm"], beta_R)
pos_main = float((res_main > 0).mean())
pos_split = float((res_split > 0).mean())
print(f"pos_label_fraction main={pos_main:.4f}  split={pos_split:.4f}")

print(f"\n=== Probe train+eval, {args.seeds} seeds [{elapsed()}] ===")
seeds = list(range(43, 43 + args.seeds))
seed_main, seed_split = [], []
for s in seeds:
    h_main, _ = train_probe(T_data, beta_T, s, TRAIN_DEVICE)
    rho_main = evaluate_head(h_main, V_data)
    h_split, _ = train_probe(T_data, beta_R, s, TRAIN_DEVICE)
    rho_split = evaluate_head(h_split, V_data)
    seed_main.append(rho_main)
    seed_split.append(rho_split)
    print(f"  seed {s}: main={rho_main:+.4f}  split={rho_split:+.4f}  delta={rho_split - rho_main:+.4f}")

mean_main = float(np.mean(seed_main))
mean_split = float(np.mean(seed_split))
delta = mean_split - mean_main

output = {
    "model": args.model,
    "n_layers": int(N_LAYERS),
    "hidden_dim": int(HIDDEN_DIM),
    "layer": int(args.layer),
    "provenance": {
        "model_revision": model_revision,
        "script": "scripts/run_residualizer_split.py",
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "value_source": "runtime",
        "device": str(DEVICE),
    },
    "protocol": {
        "ex_per_dim": int(args.ex_dim),
        "batch_size": int(args.batch_size),
        "n_seeds": int(args.seeds),
        "eval_seeds": seeds,
        "n_tokens_T": int(len(T_data["losses"])),
        "n_tokens_R": int(len(R_data["losses"])),
        "n_tokens_V": int(len(V_data["losses"])),
        "n_docs_T": int(len(T_docs)),
        "n_docs_R": int(len(R_docs)),
        "n_docs_V": int(len(val_docs)),
    },
    "residualizer_beta_main": beta_T.tolist(),
    "residualizer_beta_split": beta_R.tolist(),
    "main": {
        "mean": mean_main,
        "std": float(np.std(seed_main)),
        "per_seed": seed_main,
        "pos_label_fraction": pos_main,
    },
    "split_fit": {
        "mean": mean_split,
        "std": float(np.std(seed_split)),
        "per_seed": seed_split,
        "pos_label_fraction": pos_split,
    },
    "delta_rho": delta,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {OUT_PATH}")
print(f"main:   {mean_main:+.4f}")
print(f"split:  {mean_split:+.4f}")
print(f"delta:  {delta:+.4f}")
print(f"pos_main:  {pos_main:.4f}")
print(f"pos_split: {pos_split:.4f}")
print(f"Total: {elapsed()}")
