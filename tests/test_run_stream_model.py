"""Static checks on run_stream_model.py and split_bootstrap_gpu.py.

Catches the class of bugs that surfaced in the v4.0.0 audit:
  - Streaming script silently used latest hub HEAD instead of pinned commit.
  - Resume-stage label written by save_checkpoint did not match resume checks,
    forcing recompute on restart.
  - device_map="auto" path indexed across mismatched devices (shift_mask on
    input device, logits/hidden states on output device).
  - Bootstrap script hard-coded model.model.layers, failing on GPT-2 / Pythia.

Each test is a static parse so we can lock the invariants without GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAM = REPO_ROOT / "scripts" / "run_stream_model.py"
BOOTSTRAP = REPO_ROOT / "scripts" / "split_bootstrap_gpu.py"


# ── run_stream_model.py: revision pinning ────────────────────────────


def test_stream_model_passes_revision_to_loaders():
    """Both AutoTokenizer.from_pretrained and AutoModelForCausalLM.from_pretrained
    must receive the manifest-pinned revision (directly or via **kwargs)."""
    src = STREAM.read_text()

    assert "_revision_kwargs" in src, "run_stream_model.py must define _revision_kwargs helper"
    assert re.search(r"_rev_kw\s*=\s*_revision_kwargs\(", src), (
        "run_stream_model.py must call _revision_kwargs(MODEL_ID) at startup"
    )

    tok_call = re.search(r"AutoTokenizer\.from_pretrained\((.*?)\)", src, flags=re.DOTALL)
    assert tok_call, "AutoTokenizer.from_pretrained call not found"
    assert "_rev_kw" in tok_call.group(1) or "revision=" in tok_call.group(1), (
        "AutoTokenizer.from_pretrained must receive revision (via **_rev_kw or revision=)"
    )

    model_calls = re.findall(r"AutoModelForCausalLM\.from_pretrained\((.*?)\)", src, flags=re.DOTALL)
    assert model_calls, "AutoModelForCausalLM.from_pretrained call(s) not found"
    # load_kwargs must include _rev_kw merged in.
    assert re.search(r"load_kwargs\s*=\s*\{[^}]*\*\*_rev_kw[^}]*\}", src), (
        "load_kwargs must merge **_rev_kw so revision flows into both single-GPU and device_map=auto loaders"
    )


# ── run_stream_model.py: resume stage round-trip ─────────────────────


def _written_stages(src: str) -> set[str]:
    """Stage labels written by save_checkpoint(STAGE, ...) calls."""
    return set(re.findall(r'save_checkpoint\(\s*"([^"]+)"', src))


def _recognised_stages(src: str) -> set[str]:
    """Stage labels the resume logic checks for via _checkpoint_stage."""
    stages: set[str] = set()
    for m in re.finditer(r'_checkpoint_stage[^"\']*?"([^"]+)"', src):
        stages.add(m.group(1))
    for m in re.finditer(r"in\s*\(([^)]*)\)", src):
        for s in re.findall(r'"([^"]+)"', m.group(1)):
            stages.add(s)
    return stages


def test_stream_model_resume_stages_round_trip():
    """Every stage written by save_checkpoint must be a stage the resume
    branches recognise; otherwise the script redoes work on restart."""
    src = STREAM.read_text()
    written = _written_stages(src)
    recognised = _recognised_stages(src)

    assert written, "Found no save_checkpoint(STAGE, ...) calls"
    unknown = written - recognised
    assert not unknown, (
        f"save_checkpoint writes stages {unknown} that no resume check recognises. "
        f"On restart, the script will redo work past the last unrecognised stage. "
        f"Recognised stages: {sorted(recognised)}"
    )


# ── run_stream_model.py: device_map="auto" cross-device casts ────────


def test_stream_collector_casts_shift_mask_per_device():
    """Under device_map=auto, shift_mask (built from attn_mask on input device)
    must be cast to the consumer's device before boolean indexing into tensors
    that may live on a different GPU. Locks the post-fix structure."""
    src = STREAM.read_text()

    # Mask must be cast to the logits' device for losses_2d/sm_2d/ent_2d indexing.
    assert re.search(r"shift_mask\.to\(\s*shift_logits\.device", src), (
        "shift_mask must be cast to shift_logits.device before indexing logits-derived tensors"
    )
    # Mask must also be cast to each layer's hidden-state device for h indexing.
    assert re.search(r"shift_mask\.to\(\s*h\.device", src), (
        "shift_mask must be cast to h.device before indexing the hooked-layer activations"
    )
    # Labels go to logits' device too (cross_entropy targets must match input device).
    assert re.search(r"input_ids\[\s*:\s*,\s*1\s*:\s*\]\.to\(\s*shift_logits\.device", src), (
        "shift_labels must be cast to shift_logits.device before cross_entropy"
    )


# ── split_bootstrap_gpu.py: architecture-agnostic layer dispatch ─────


def test_bootstrap_uses_get_layer_list():
    """Bootstrap must not hard-code model.model.layers; it must dispatch
    through _get_layer_list so GPT-2 / Pythia / Gemma 3 wrappers work."""
    src = BOOTSTRAP.read_text()

    assert "def _get_layer_list" in src, "split_bootstrap_gpu.py must define _get_layer_list"

    # No bare model.model.layers[...] indexing outside _get_layer_list itself.
    bare_access = [line for line in src.splitlines() if "model.model.layers[" in line]
    assert not bare_access, (
        "split_bootstrap_gpu.py still indexes model.model.layers[...] directly: "
        f"{bare_access}. Use _get_layer_list(model)[layer] for architecture parity."
    )

    # Hook registration must dispatch via _get_layer_list.
    hook_lines = [line for line in src.splitlines() if "register_forward_hook" in line]
    assert hook_lines, "Could not find a register_forward_hook call"
    assert any("_get_layer_list(" in line for line in hook_lines), (
        f"Hook registration must go through _get_layer_list(model)[layer]; got: {hook_lines}"
    )


def _qwen_like():
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    return m


def _gpt2_like():
    m = nn.Module()
    m.transformer = nn.Module()
    m.transformer.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    return m


def _pythia_like():
    m = nn.Module()
    m.gpt_neox = nn.Module()
    m.gpt_neox.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    return m


def _unsupported():
    m = nn.Module()
    m.foo = nn.Linear(4, 4)
    return m


def _extract_function(filepath: Path, func_name: str) -> str:
    lines = filepath.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
            break
    if start is None:
        pytest.fail(f"{func_name} not found in {filepath.name}")
    func_lines = [lines[start]]
    for line in lines[start + 1 :]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def _compile_function(source: str, name: str):
    ns = {"torch": torch, "nn": nn}
    exec(compile(source, f"<{name}>", "exec"), ns)
    return ns[name]


def test_bootstrap_get_layer_list_resolves_each_family():
    """Resolve Llama/Qwen-shaped, GPT-2-shaped, and Pythia-shaped models;
    raise on unknown shapes."""
    fn = _compile_function(_extract_function(BOOTSTRAP, "_get_layer_list"), "_get_layer_list")

    qwen = _qwen_like()
    assert fn(qwen) is qwen.model.layers
    gpt2 = _gpt2_like()
    assert fn(gpt2) is gpt2.transformer.h
    pythia = _pythia_like()
    assert fn(pythia) is pythia.gpt_neox.layers

    with pytest.raises(ValueError, match="Unsupported architecture"):
        fn(_unsupported())
