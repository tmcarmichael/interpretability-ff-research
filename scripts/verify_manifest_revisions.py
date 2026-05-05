"""Verify each commit in results/model_revisions.json exists on Hugging Face.

Queries the Hugging Face API for every entry in the manifest, confirms the
commit hash is a valid revision of the corresponding repo, and emits a
timestamped JSON report at results/manifest_verification/<date>.json.

Output entries record one of:
    "verified": HF API returned the same commit hash
    "matches_repo_only": commit exists but HF API returned a different hash
                         (e.g., manifest pins an old commit; the repo HEAD has
                         moved on, but the pinned commit is still resolvable)
    "gated_no_access": gated repo without HF_TOKEN access; not counted as failure
    "repo_not_found": Hugging Face repo does not exist at the given id
    "revision_not_found": repo exists but the commit hash is not a valid revision
    "error: <type>": other transport or API errors

Usage:
    HF_TOKEN=hf_xxx uv run --extra transformer python scripts/verify_manifest_revisions.py
    (HF_TOKEN optional; only required for gated repos)
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST = RESULTS_DIR / "model_revisions.json"


def main():
    if not MANIFEST.is_file():
        sys.exit(f"Manifest missing: {MANIFEST}")

    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["models"]
    api = HfApi()

    results = []
    n_verified = 0
    n_gated = 0
    n_failures = 0

    for model_id, entry in entries.items():
        commit = entry["commit"]
        record = {"model_id": model_id, "manifest_commit": commit}

        try:
            info = api.repo_info(model_id, revision=commit)
            resolved = getattr(info, "sha", None) or commit
            record["hf_resolved_commit"] = resolved
            if resolved == commit:
                record["status"] = "verified"
                n_verified += 1
            else:
                record["status"] = "matches_repo_only"
                n_verified += 1
        except RepositoryNotFoundError:
            record["status"] = "repo_not_found"
            n_failures += 1
        except RevisionNotFoundError:
            record["status"] = "revision_not_found"
            n_failures += 1
        except GatedRepoError:
            record["status"] = "gated_no_access"
            n_gated += 1
        except Exception as e:
            record["status"] = f"error: {type(e).__name__}: {e}"
            n_failures += 1

        print(f"  {record['status']:22s}  {model_id:50s}  {commit[:12]}")
        results.append(record)

    now = datetime.now(UTC)
    today = now.strftime("%Y-%m-%d")
    out_dir = RESULTS_DIR / "manifest_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}.json"

    output = {
        "verified_at": now.isoformat(timespec="seconds"),
        "manifest_path": "results/model_revisions.json",
        "manifest_retrieved": manifest.get("retrieved"),
        "n_entries": len(entries),
        "n_verified": n_verified,
        "n_gated_no_access": n_gated,
        "n_failures": n_failures,
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))

    print()
    print(f"Wrote {out_path}")
    print(f"Verified: {n_verified}/{len(entries)}")
    print(f"Gated (auth required, not a failure): {n_gated}")
    print(f"Failures: {n_failures}")

    if n_failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
