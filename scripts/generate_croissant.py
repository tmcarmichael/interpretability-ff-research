"""Generate croissant.json metadata descriptor from results/ and schema/.

Builds a Croissant 1.1 metadata descriptor (Core + recordSet) per the
official spec at https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html.

Sources of truth:
  - pyproject.toml: name, version, description, keywords, license
  - CITATION.cff: citation text, DOI, date
  - LICENSE: SPDX license name
  - results/*.json: distribution file inventory + sha256
  - results/manifest_verification/*.json: latest verification report
  - schema/*.schema.json: per-record-type field definitions

Usage:
  uv run python scripts/generate_croissant.py            # write croissant.json
  uv run python scripts/generate_croissant.py --check    # diff against committed file
  uv run python scripts/generate_croissant.py --quiet    # suppress progress lines
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCHEMA = REPO / "schema"
OUT = REPO / "croissant.json"

# Map filename suffix -> (file-type id, schema name, human description).
# Mirrors scripts/validate_schemas.py DISPATCH so a new file type added there
# is visible here too.
FILE_TYPES: list[tuple[str, str, str, str]] = [
    (
        "_main.json",
        "main-results",
        "main",
        "Per-model output of the canonical 350 ex/dim 7-seed observability protocol.",
    ),
    (
        "_nonlinear-probe-multilayer.json",
        "nonlinear-probe-multilayer-results",
        "nonlinear-probe",
        "Multi-layer nonlinear-probe sweep on Llama 3B.",
    ),
    (
        "_nonlinear-probe.json",
        "nonlinear-probe-results",
        "nonlinear-probe",
        "Matched-HP and swept-HP MLP-vs-linear comparison per model.",
    ),
    (
        "_dynamics.json",
        "dynamics-results",
        "dynamics",
        "Pythia checkpoint trajectory: pcorr and oc_residual across training steps.",
    ),
    (
        "_residualizer-split.json",
        "residualizer-split-results",
        "residualizer-split",
        "Split-fit OLS residualizer test (separate fit pool from probe pool).",
    ),
    (
        "_mechanistic.json",
        "mechanistic-results",
        "mechanistic",
        "Mean-ablation patching by layer/component (sign-only interpretation).",
    ),
    (
        "_squad-rag.json",
        "squad-results",
        "downstream",
        "SQuAD 2.0 reading-comprehension exclusive-catch rates by flag rate.",
    ),
    (
        "_medqa.json",
        "medqa-results",
        "downstream",
        "MedQA-USMLE multiple-choice exclusive-catch rates by flag rate.",
    ),
    (
        "_truthfulqa.json",
        "truthfulqa-results",
        "downstream",
        "TruthfulQA generation exclusive-catch rates and AUC by flag rate.",
    ),
    (
        "_shuffle-control.json",
        "shuffle-control-results",
        "shuffle-control",
        "Shuffled-label probe null distribution (10 permutations).",
    ),
    (
        "_bootstrap.json",
        "bootstrap-results",
        "bootstrap",
        "Document-level bootstrap on Qwen 7B (30 resamples).",
    ),
    (
        "_width-sweep.json",
        "width-sweep-results",
        "width-sweep",
        "Output-side MLP width sweep (64-512 units) on Qwen 7B.",
    ),
]
LEGACY_PATTERNS = (
    "_sae-comparison.json",
    "_bottleneck-scaling.json",
    "_exdim-1000.json",
    "_exdim-sweep.json",
    "transformer_observe.json",
)
SKIP = {"model_revisions.json", "dataset_revisions.json", "figure_sources.json"}

# JSON Schema "type" -> Croissant dataType IRI.
TYPE_MAP = {
    "string": "sc:Text",
    "integer": "sc:Integer",
    "number": "sc:Float",
    "boolean": "sc:Boolean",
    "array": "sc:Text",  # arrays serialized inline
    "object": "sc:Text",  # nested objects serialized inline
}

# Croissant 1.1 Appendix 1 with one URL-scheme adjustment: schema.org is
# referenced as https://schema.org/ rather than http://. RDF treats them as
# the same vocabulary, but the mlcroissant validator (built before the 1.1
# spec landed) does strict string matching on Dataset URI and rejects http://.
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
}

CONFORMS_TO = "http://mlcommons.org/croissant/1.1"
SD_VERSION = "1.1"


def _read_pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]


def _read_citation() -> dict:
    text = (REPO / "CITATION.cff").read_text()
    fields = {}
    for line in text.splitlines():
        m = re.match(r'^(version|date-released|doi|repository-code|title): "?([^"]+?)"?$', line)
        if m:
            fields[m.group(1)] = m.group(2)
    return fields


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(path: Path) -> tuple[str, str, str, str] | None:
    """Return (fileset_id, schema_name, description, suffix) for a result file, or None."""
    name = path.name
    if name in SKIP or name.startswith("."):
        return None
    for suffix, fileset_id, schema_name, desc in FILE_TYPES:
        if name.endswith(suffix):
            return fileset_id, schema_name, desc, suffix
    if any(name.endswith(p) for p in LEGACY_PATTERNS):
        return (
            "legacy-results",
            "legacy",
            "Pre-v3.0.0 phase-taxonomy outputs preserved as provenance.",
            "legacy",
        )
    return None


def _load_schema(name: str) -> dict | None:
    path = SCHEMA / f"{name}.schema.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _schema_to_fields(schema: dict, fileset_id: str) -> list[dict]:
    """Map JSON Schema top-level properties to Croissant Field entries."""
    properties = schema.get("properties", {})
    fields = []
    for name, spec in properties.items():
        js_type = spec.get("type", "string")
        if isinstance(js_type, list):
            js_type = next((t for t in js_type if t != "null"), "string")
        data_type = TYPE_MAP.get(js_type, "sc:Text")
        field = {
            "@type": "cr:Field",
            "@id": f"{fileset_id}-records/{name}",
            "name": name,
            "description": spec.get("description", "").strip() or f"Field {name} from {fileset_id}.",
            "dataType": data_type,
            "source": {
                "fileSet": {"@id": fileset_id},
                "extract": {"jsonPath": f"$.{name}"},
            },
        }
        if js_type in ("array", "object"):
            field["description"] = (field["description"] + " (JSON-serialized)").strip()
        fields.append(field)
    return fields


def _archive_sha256(grouped: dict[str, list[Path]]) -> str:
    """Deterministic merkle hash over (filename, sha256) pairs of every distribution file.

    Lets a reviewer verify dataset integrity without downloading a tarball:
    rerun the generator on a clean clone and the same hash should appear.
    """
    pairs: list[tuple[str, str]] = []
    for paths in grouped.values():
        for p in paths:
            pairs.append((p.name, _sha256(p)))
    for special in ("model_revisions.json", "dataset_revisions.json"):
        pairs.append((special, _sha256(RESULTS / special)))
    for p in sorted((RESULTS / "manifest_verification").glob("*.json")):
        pairs.append((p.name, _sha256(p)))
    h = hashlib.sha256()
    for name, sha in sorted(pairs):
        h.update(f"{name}:{sha}\n".encode())
    return h.hexdigest()


def _file_objects(citation: dict, grouped: dict[str, list[Path]]) -> list[dict]:
    """Parent archive + singleton FileObjects for the pinned manifests."""
    repo_url = citation["repository-code"].rstrip("/")
    version = citation["version"]
    manifests = sorted((RESULTS / "manifest_verification").glob("*.json"))
    latest = manifests[-1] if manifests else None
    out = [
        {
            "@type": "cr:FileObject",
            "@id": "nn-observability-archive",
            "name": "nn-observability-archive",
            "description": (
                f"Source repository at v{version}. The sha256 is a merkle hash over "
                "(filename, sha256) pairs of every distribution file under results/, "
                "computed by scripts/generate_croissant.py; rerun the generator on a "
                "clean clone to verify."
            ),
            "contentUrl": f"{repo_url}/archive/refs/tags/v{version}.tar.gz",
            "encodingFormat": "application/x-tar",
            "sha256": _archive_sha256(grouped),
        },
        {
            "@type": "cr:FileObject",
            "@id": "model-revisions",
            "name": "model-revisions",
            "description": "Hugging Face model IDs and pinned commit hashes for every evaluated model.",
            "containedIn": {"@id": "nn-observability-archive"},
            "contentUrl": "results/model_revisions.json",
            "encodingFormat": "application/json",
            "sha256": _sha256(RESULTS / "model_revisions.json"),
        },
        {
            "@type": "cr:FileObject",
            "@id": "dataset-revisions",
            "name": "dataset-revisions",
            "description": "Hugging Face dataset IDs and pinned commit hashes for every "
            "paper-cited evaluation corpus.",
            "containedIn": {"@id": "nn-observability-archive"},
            "contentUrl": "results/dataset_revisions.json",
            "encodingFormat": "application/json",
            "sha256": _sha256(RESULTS / "dataset_revisions.json"),
        },
    ]
    if latest is not None:
        out.append(
            {
                "@type": "cr:FileObject",
                "@id": "manifest-verification",
                "name": "manifest-verification",
                "description": "Programmatic verification of every model_revisions entry "
                "against the live Hugging Face API.",
                "containedIn": {"@id": "nn-observability-archive"},
                "contentUrl": f"results/manifest_verification/{latest.name}",
                "encodingFormat": "application/json",
                "sha256": _sha256(latest),
            }
        )
    return out


def _file_sets(grouped: dict[str, list[Path]], fileset_meta: dict[str, str]) -> list[dict]:
    """One FileSet per file-type group, each containedIn the parent archive."""
    out = []
    for fileset_id in sorted(grouped):
        out.append(
            {
                "@type": "cr:FileSet",
                "@id": fileset_id,
                "name": fileset_id,
                "description": fileset_meta[fileset_id],
                "containedIn": {"@id": "nn-observability-archive"},
                "encodingFormat": "application/json",
                "includes": [f"results/{p.name}" for p in sorted(grouped[fileset_id])],
            }
        )
    return out


def _record_sets(grouped: dict[str, list[Path]], schemas: dict[str, str]) -> list[dict]:
    """One RecordSet per file type, fields derived from the matching JSON Schema."""
    out = []
    for fileset_id in sorted(grouped):
        schema_name = schemas[fileset_id]
        schema = _load_schema(schema_name)
        if schema is None:
            continue
        out.append(
            {
                "@type": "cr:RecordSet",
                "@id": f"{fileset_id}-records",
                "name": f"{fileset_id}-records",
                "description": (schema.get("description") or f"Records derived from {fileset_id}."),
                "key": {"@id": f"{fileset_id}-records/model"}
                if "model" in schema.get("properties", {})
                else None,
                "field": _schema_to_fields(schema, fileset_id),
            }
        )
        # Drop the optional "key" if there's no "model" field.
        if out[-1]["key"] is None:
            del out[-1]["key"]
    return out


def _walk_results() -> tuple[dict[str, list[Path]], dict[str, str], dict[str, str]]:
    """Group results files by file-type id. Return (groups, descriptions, schemas)."""
    groups: dict[str, list[Path]] = {}
    descriptions: dict[str, str] = {}
    schemas: dict[str, str] = {}
    unmatched: list[Path] = []
    for p in sorted(RESULTS.glob("*.json")):
        info = _classify(p)
        if info is None:
            if p.name not in SKIP:
                unmatched.append(p)
            continue
        fileset_id, schema_name, desc, _ = info
        groups.setdefault(fileset_id, []).append(p)
        descriptions[fileset_id] = desc
        schemas[fileset_id] = schema_name
    if unmatched:
        names = ", ".join(p.name for p in unmatched)
        sys.exit(f"FAIL: unmatched result file(s): {names}")
    return groups, descriptions, schemas


def build() -> dict:
    pyproj = _read_pyproject()
    citation = _read_citation()
    groups, descriptions, schemas = _walk_results()

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CONFORMS_TO,
        "sdVersion": SD_VERSION,
        "name": pyproj["name"],
        "description": pyproj["description"],
        "url": citation["repository-code"],
        "version": citation["version"],
        "datePublished": citation["date-released"],
        "license": "https://opensource.org/licenses/MIT",
        "citeAs": (
            f"Carmichael, T. ({citation['date-released'][:4]}). "
            f"{citation['title']}. Zenodo. https://doi.org/{citation['doi']}"
        ),
        "creator": {"@type": "Person", "name": "Thomas Carmichael"},
        "keywords": pyproj["keywords"],
        "sameAs": f"https://doi.org/{citation['doi']}",
        "isLiveDataset": False,
        "distribution": _file_objects(citation, groups) + _file_sets(groups, descriptions),
        "recordSet": _record_sets(groups, schemas),
    }


def _emit(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check", action="store_true", help="Diff regenerated content against committed croissant.json."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args(argv)

    text = _emit(build())

    if args.check:
        if not OUT.is_file():
            print("FAIL: croissant.json missing; run without --check to generate.")
            return 1
        committed = OUT.read_text()
        if committed != text:
            print("FAIL: croissant.json out of date. Re-run scripts/generate_croissant.py.")
            return 1
        if not args.quiet:
            print(f"OK: croissant.json matches generator (len={len(text)} bytes).")
        return 0

    OUT.write_text(text)
    if not args.quiet:
        n_files = sum(len(v) for v in _walk_results()[0].values())
        print(f"Wrote {OUT.relative_to(REPO)} ({len(text)} bytes, {n_files} result files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
