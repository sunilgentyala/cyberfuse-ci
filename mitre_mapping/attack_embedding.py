"""
mitre_mapping/attack_embedding.py

Builds the FAISS embedding index from MITRE ATT&CK v15 STIX data.

Downloads ATT&CK STIX from the MITRE CTI GitHub repository, encodes
each technique description using the all-MiniLM-L6-v2 sentence transformer,
and writes a FAISS IndexFlatIP (inner product = cosine similarity for
normalized vectors) plus metadata and mitigation lookup JSON files.

Run this once before using technique_mapper.py.

Usage:
    python mitre_mapping/attack_embedding.py --output mitre_mapping/index/

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)
MODEL_NAME = "all-MiniLM-L6-v2"


def download_attack_stix(url: str = ATTACK_STIX_URL) -> dict:
    logger.info("Downloading ATT&CK STIX bundle from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_techniques(bundle: dict) -> tuple[list[dict], dict]:
    """
    Extract technique metadata and mitigation relationships.

    Returns:
        techniques:     List of dicts with technique_id, name, tactic, description
        mit_map:        Dict mapping technique_id to mitigation info
    """
    objects_by_id = {obj["id"]: obj for obj in bundle.get("objects", [])}

    techniques = []
    mit_map: dict = {}

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_deprecated") or obj.get("revoked"):
            continue

        tid = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id", "")
                break

        if not tid:
            continue

        tactic = ""
        kill_chain_phases = obj.get("kill_chain_phases", [])
        if kill_chain_phases:
            tactic = kill_chain_phases[0].get("phase_name", "").replace("-", " ").title()

        description = obj.get("description", obj.get("name", ""))

        techniques.append({
            "stix_id":      obj["id"],
            "technique_id": tid,
            "name":         obj.get("name", ""),
            "tactic":       tactic,
            "description":  description,
        })

    # Extract mitigation relationships
    for obj in bundle.get("objects", []):
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "mitigates":
            continue

        mit_stix_id  = obj.get("source_ref", "")
        tech_stix_id = obj.get("target_ref", "")
        mit_obj      = objects_by_id.get(mit_stix_id, {})

        mid = ""
        for ref in mit_obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                mid = ref.get("external_id", "")
                break

        # Resolve technique_id for the target
        tech_obj = objects_by_id.get(tech_stix_id, {})
        tech_id = ""
        for ref in tech_obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id", "")
                break

        if tech_id and mid:
            mit_map[tech_id] = {
                "mitigation_id":   mid,
                "mitigation_name": mit_obj.get("name", ""),
            }

    return techniques, mit_map


def build_index(output_dir: str = "mitre_mapping/index"):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Install faiss and sentence-transformers:\n"
            "  pip install faiss-cpu sentence-transformers"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = download_attack_stix()
    techniques, mit_map = extract_techniques(bundle)

    logger.info("Encoding %d ATT&CK techniques with %s", len(techniques), MODEL_NAME)
    encoder = SentenceTransformer(MODEL_NAME)
    descriptions = [t["description"][:512] for t in techniques]
    embeddings = encoder.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out / "index.faiss"))
    logger.info("FAISS index written with %d vectors (dim=%d)", index.ntotal, dim)

    metadata = [{k: v for k, v in t.items() if k != "description"} for t in techniques]
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with (out / "mitigation_map.json").open("w", encoding="utf-8") as f:
        json.dump(mit_map, f, ensure_ascii=False, indent=2)

    logger.info(
        "ATT&CK index built. %d techniques indexed. %d mitigation mappings saved.",
        len(techniques), len(mit_map),
    )


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build MITRE ATT&CK FAISS embedding index")
    parser.add_argument("--output", default="mitre_mapping/index/", help="Output directory for index files")
    args = parser.parse_args()
    build_index(output_dir=args.output)


if __name__ == "__main__":
    main()
