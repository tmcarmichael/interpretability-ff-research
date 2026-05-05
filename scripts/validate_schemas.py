"""Validate result JSONs against the formal JSON Schemas in schema/.

Walks every results/*.json (excluding manifests) and validates each against
the schema matched by filename pattern. Reports per-file pass/fail.

Usage:
    uv run python scripts/validate_schemas.py
    uv run python scripts/validate_schemas.py --strict   # exit 1 on any failure
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("FAIL: jsonschema package required. Install with 'uv add jsonschema --dev'.")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCHEMA = REPO / "schema"

# Filename suffix or exact name -> schema basename. Order matters: longer
# suffixes must come first so e.g. "_nonlinear-probe-multilayer.json" is
# matched by "_nonlinear-probe" (any suffix containing "nonlinear-probe").
DISPATCH: list[tuple[str, str]] = [
    ("_main.json", "main"),
    ("_dynamics.json", "dynamics"),
    ("_residualizer-split.json", "residualizer-split"),
    ("_nonlinear-probe-multilayer.json", "nonlinear-probe"),
    ("_nonlinear-probe.json", "nonlinear-probe"),
    ("_mechanistic.json", "mechanistic"),
    ("_squad-rag.json", "downstream"),
    ("_medqa.json", "downstream"),
    ("_truthfulqa.json", "downstream"),
    ("_shuffle-control.json", "shuffle-control"),
    ("_bootstrap.json", "bootstrap"),
    ("_width-sweep.json", "width-sweep"),
    # Legacy / phase-taxonomy artifacts (provenance-only floor)
    ("_sae-comparison.json", "legacy"),
    ("_bottleneck-scaling.json", "legacy"),
    ("_exdim-1000.json", "legacy"),
    ("_exdim-sweep.json", "legacy"),
    ("transformer_observe.json", "legacy"),
]

# Files that are not result data and must be skipped.
SKIP_PATTERNS = (
    "model_revisions.json",
    "dataset_revisions.json",
    "figure_sources.json",
)


def _classify(path: Path) -> str | None:
    name = path.name
    if name in SKIP_PATTERNS:
        return None
    for suffix, schema_name in DISPATCH:
        if name.endswith(suffix):
            return schema_name
    return None


def _load_schema(name: str) -> dict:
    path = SCHEMA / f"{name}.schema.json"
    if not path.is_file():
        sys.exit(f"FAIL: schema {path} not found")
    return json.loads(path.read_text())


def _validate(file: Path, schema: dict) -> list[str]:
    """Return a list of error messages (empty if valid)."""
    try:
        data = json.loads(file.read_text())
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(data)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 on any validation failure or unmatched file."
    )
    args = parser.parse_args()

    schemas: dict[str, dict] = {}
    counts: dict[str, int] = {}
    fails: list[tuple[Path, list[str]]] = []
    unmatched: list[Path] = []

    for f in sorted(RESULTS.glob("*.json")):
        kind = _classify(f)
        if kind is None:
            if f.name not in SKIP_PATTERNS:
                unmatched.append(f)
            continue
        if kind not in schemas:
            schemas[kind] = _load_schema(kind)
        errs = _validate(f, schemas[kind])
        counts[kind] = counts.get(kind, 0) + 1
        if errs:
            fails.append((f, errs))

    n_total = sum(counts.values())
    n_pass = n_total - len(fails)
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    print(f"Schema validation: {n_pass}/{n_total} pass ({summary})")

    if fails:
        for f, errs in fails:
            print(f"\n  FAIL {f.name}:")
            for e in errs[:5]:
                print(f"    - {e}")
            if len(errs) > 5:
                print(f"    ...and {len(errs) - 5} more errors")

    if unmatched:
        print(f"\n  UNMATCHED ({len(unmatched)} file(s) have no schema dispatch):")
        for f in unmatched:
            print(f"    - {f.name}")

    if args.strict and (fails or unmatched):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
