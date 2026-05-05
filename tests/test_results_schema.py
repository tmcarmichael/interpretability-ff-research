"""Schema validation for primary results JSONs.

Catches missing fields, wrong types, and structural changes before
they propagate to downstream consumers. Parametrized over every primary
result file so a new model with a missing field fails immediately.

Run: uv run pytest tests/test_results_schema.py -v
"""

import json
import re
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_PRIMARY_NAMES = [
    "qwen2.5-0.5b_main.json",
    "qwen2.5-0.5b-instruct_main.json",
    "qwen2.5-1.5b_main.json",
    "qwen2.5-1.5b-instruct_main.json",
    "qwen2.5-3b_main.json",
    "qwen2.5-3b-instruct_main.json",
    "qwen2.5-7b_main.json",
    "qwen2.5-7b-instruct_main.json",
    "qwen2.5-14b_main.json",
    "qwen2.5-14b-instruct_main.json",
    "gemma-3-1b_main.json",
    "gemma-3-4b_main.json",
    "llama-3.2-1b_main.json",
    "llama-3.2-1b-instruct_main.json",
    "llama-3.2-3b_main.json",
    "llama-3.1-8b_main.json",
    "mistral-7b-v0.3_main.json",
    "mistral-7b-instruct-v0.3_main.json",
    "phi-3-mini_main.json",
    "pythia-70m_main.json",
    "pythia-160m_main.json",
    "pythia-410m_main.json",
    "pythia-1b_main.json",
    "pythia-1.4b_main.json",
    "pythia-1.4b-deduped_main.json",
    "pythia-2.8b_main.json",
    "pythia-6.9b_main.json",
    "pythia-12b_main.json",
]
PRIMARY_FILES = sorted(p for p in (RESULTS_DIR / n for n in _PRIMARY_NAMES) if p.exists())


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(params=PRIMARY_FILES, ids=lambda p: p.name)
def result(request):
    return _load(request.param)


def test_model_metadata(result):
    """Every result must identify the model and architecture."""
    assert "model" in result
    assert "n_layers" in result
    assert "hidden_dim" in result
    assert isinstance(result["n_layers"], int)
    assert isinstance(result["hidden_dim"], int)


def test_partial_corr_structure(result):
    """partial_corr must have mean, per_seed, and n_seeds."""
    pc = result["partial_corr"]
    assert "mean" in pc, "missing partial_corr.mean"
    assert "per_seed" in pc, "missing partial_corr.per_seed"
    assert "n_seeds" in pc, "missing partial_corr.n_seeds"
    assert isinstance(pc["mean"], (int, float))
    assert isinstance(pc["per_seed"], list)
    assert len(pc["per_seed"]) == pc["n_seeds"]
    assert len(pc["per_seed"]) >= 3, f"only {len(pc['per_seed'])} seeds"


def test_partial_corr_range(result):
    """pcorr should be between -1 and +1."""
    pc = result["partial_corr"]
    assert -1 <= pc["mean"] <= 1
    for val in pc["per_seed"]:
        assert -1 <= val <= 1


def test_output_controlled(result):
    """output_controlled must have mean."""
    oc = result["output_controlled"]
    assert "mean" in oc
    assert isinstance(oc["mean"], (int, float))


def test_seed_agreement(result):
    """seed_agreement must have mean >= 0."""
    sa = result["seed_agreement"]
    assert isinstance(sa, dict), "seed_agreement should be a dict"
    assert "mean" in sa
    assert sa["mean"] >= 0


def test_peak_layer(result):
    """Peak layer must exist and be within layer range."""
    peak = result.get("peak_layer_final", result.get("peak_layer"))
    assert peak is not None, "missing peak_layer_final or peak_layer"
    assert 0 <= peak < result["n_layers"]
    assert "peak_layer_frac" in result
    assert 0 <= result["peak_layer_frac"] <= 1


def test_baselines(result):
    """baselines dict must exist with at least one entry."""
    assert "baselines" in result
    assert len(result["baselines"]) > 0


def test_control_sensitivity(result):
    """control_sensitivity must have the standard control set."""
    cs = result["control_sensitivity"]
    if "_incomplete" in cs:
        pytest.skip("control_sensitivity incomplete (partial result)")
    for key in ("none", "softmax_only", "standard"):
        assert key in cs, f"missing control_sensitivity.{key}"
        assert isinstance(cs[key], (int, float))


def test_layer_profile(result):
    """layer_profile must exist with at least 3 entries."""
    lp = result["layer_profile"]
    assert isinstance(lp, dict)
    assert len(lp) >= 3, f"layer_profile has only {len(lp)} entries"


def test_protocol(result):
    """protocol must document the evaluation setup."""
    proto = result["protocol"]
    assert "eval_seeds" in proto or "layer_select_seed" in proto


def test_canonical_provenance_all_files():
    """Every committed result JSON conforms to the canonical 5-field shape.

    Wires `validate_canonical_provenance` into the test suite so any future
    commit that introduces a non-CUDA device, microsecond timestamp, short
    SHA, missing value_source, or out-of-order field set fails CI.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.load_results import validate_canonical_provenance

    n_warnings = validate_canonical_provenance()
    assert n_warnings == 0, f"{n_warnings} canonical-provenance violations (see stdout)"


_GOOD_PROVENANCE = {
    "model_revision": "0" * 40,
    "script": "scripts/run_model.py",
    "timestamp": "2026-05-02T04:36:54+00:00",
    "value_source": "runtime",
    "device": "cuda",
}


_MUTATIONS = [
    ("device_mps", {**_GOOD_PROVENANCE, "device": "mps"}),
    ("device_cpu", {**_GOOD_PROVENANCE, "device": "cpu"}),
    ("script_directory", {**_GOOD_PROVENANCE, "script": "notebooks/"}),
    ("script_empty", {**_GOOD_PROVENANCE, "script": ""}),
    ("script_missing_file", {**_GOOD_PROVENANCE, "script": "scripts/does_not_exist.py"}),
    ("script_outside_tree", {**_GOOD_PROVENANCE, "script": "/etc/passwd"}),
    ("extra_field_torch_version", {**_GOOD_PROVENANCE, "torch_version": "2.8.0"}),
    ("revision_unknown", {**_GOOD_PROVENANCE, "model_revision": "unknown"}),
    ("revision_short_sha", {**_GOOD_PROVENANCE, "model_revision": "abc123"}),
    ("timestamp_microsecond", {**_GOOD_PROVENANCE, "timestamp": "2026-05-02T04:36:54.123456+00:00"}),
    ("value_source_invalid", {**_GOOD_PROVENANCE, "value_source": "made_up"}),
    ("missing_field", {k: v for k, v in _GOOD_PROVENANCE.items() if k != "device"}),
]


@pytest.mark.parametrize("label,prov", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_provenance_mutations_rejected(label, prov):
    """Each mutation should produce at least one error.

    Locks the canonical contract from drift: if a future change to the
    validator inadvertently widens what passes (e.g. re-accepts 'unknown',
    drops the script existence check), one of these mutations breaks the
    test loudly.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.load_results import _validate_one_provenance

    errors = _validate_one_provenance(prov, label)
    assert errors, f"mutation '{label}' should fail validation but passed"


def test_provenance_known_good_passes():
    """Sanity check: the unmutated dict passes the validator."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.load_results import _validate_one_provenance

    assert _validate_one_provenance(_GOOD_PROVENANCE) == []


def test_manifest_provenance_consistency():
    """Every paper-cited result's model_revision is recorded in the manifest.

    Producers read results/model_revisions.json to pin from_pretrained() to
    a SHA. The committed JSON's provenance.model_revision is what was loaded
    when the result was produced. If these diverge, a fresh re-run would load
    different weights than the committed result was produced with, breaking
    the reproducibility contract silently.

    Aggregate files (cross_family, dynamics) carry model_revision="multi" and
    are skipped. Multi-model aggregates produced by transformer_observe.py
    (transformer_observe.json, gpt-2-124m_sae-comparison.json,
    gpt-2_bottleneck-scaling.json) legitimately have model=None at the top
    level. When a result declares a single non-empty model id, that id must
    be a manifest key and the SHA must match that exact entry.
    """
    manifest = _load(RESULTS_DIR / "model_revisions.json")
    manifest_by_id = {mid: e["commit"] for mid, e in manifest["models"].items()}
    manifest_shas = set(manifest_by_id.values())

    exempt = {"model_revisions.json", "dataset_revisions.json"}
    failures = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in exempt:
            continue
        d = _load(path)
        rev = d.get("provenance", {}).get("model_revision", "")
        if rev == "multi":
            for step, ck in (d.get("checkpoints") or {}).items():
                ck_rev = ck.get("revision", "")
                if not re.fullmatch(r"[0-9a-f]{40}", ck_rev):
                    failures.append(f"{path.name}: checkpoint {step!r} revision not 40-char SHA: {ck_rev!r}")
            continue
        if rev not in manifest_shas:
            failures.append(f"{path.name}: revision {rev[:12]} not in manifest")
            continue
        model_id = d.get("model")
        if model_id:
            if model_id not in manifest_by_id:
                failures.append(f"{path.name}: model={model_id!r} not a manifest key")
            elif manifest_by_id[model_id] != rev:
                failures.append(
                    f"{path.name}: model={model_id} expects {manifest_by_id[model_id][:12]}, got {rev[:12]}"
                )
    assert not failures, "manifest/provenance drift:\n  " + "\n  ".join(failures)


def test_dataset_revisions_schema():
    """results/dataset_revisions.json: required fields, 40-char SHAs, well-formed entries."""
    p = RESULTS_DIR / "dataset_revisions.json"
    assert p.is_file(), f"dataset_revisions.json missing: {p}"
    data = _load(p)
    for top in ("description", "retrieved", "datasets"):
        assert top in data, f"dataset_revisions.json: missing top-level field {top!r}"
    assert isinstance(data["datasets"], dict) and data["datasets"], (
        "dataset_revisions.json: 'datasets' must be a non-empty dict"
    )
    for name, entry in data["datasets"].items():
        for field in ("commit", "url", "config"):
            assert field in entry, f"{name}: missing field {field!r}"
        assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"]), (
            f"{name}: commit not a 40-char hex SHA: {entry['commit']!r}"
        )
        assert entry["url"].startswith("https://huggingface.co/datasets/"), (
            f"{name}: url must start with https://huggingface.co/datasets/"
        )
        assert entry["url"].endswith(name), (
            f"{name}: url path does not match dataset key (got {entry['url']!r})"
        )
        if entry["config"] is not None:
            assert isinstance(entry["config"], str) and entry["config"], (
                f"{name}: config must be a non-empty string or null"
            )
