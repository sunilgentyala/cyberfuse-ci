"""
mitre_mapping/technique_mapper.py

Maps classified vulnerability entities to MITRE ATT&CK v15 techniques
using nearest-neighbor lookup in technique embedding space.

The embedding index is built from the STIX 2.1 representation of
ATT&CK v15 (https://github.com/mitre/cti). Each technique description
is encoded using a sentence-transformer model, then indexed with FAISS
for sub-millisecond lookup at inference time.

The mapping output per vulnerability:
    technique_id:     ATT&CK technique ID, e.g. T1190
    technique_name:   Human-readable technique name
    tactic:           ATT&CK tactic (Initial Access, Execution, etc.)
    confidence:       Cosine similarity score 0.0 to 1.0
    mitigation_id:    Recommended mitigation ID, e.g. M1016
    mitigation_name:  Human-readable mitigation name

Usage:
    # Build the embedding index (run once, then reuse the .faiss file)
    python mitre_mapping/attack_embedding.py --output mitre_mapping/index/

    # Map a single CVE
    python mitre_mapping/technique_mapper.py --cve CVE-2024-12345

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).parent / "index"
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class TechniqueMatch:
    technique_id:    str
    technique_name:  str
    tactic:          str
    confidence:      float
    mitigation_id:   Optional[str]
    mitigation_name: Optional[str]

    def to_dict(self) -> dict:
        return {
            "technique_id":    self.technique_id,
            "technique_name":  self.technique_name,
            "tactic":          self.tactic,
            "confidence":      round(self.confidence, 4),
            "mitigation_id":   self.mitigation_id,
            "mitigation_name": self.mitigation_name,
        }


class ATTACKTechniqueMapper:
    """
    Nearest-neighbor MITRE ATT&CK technique mapper.

    The mapper encodes the vulnerability description as a sentence embedding
    and finds the most similar ATT&CK technique by cosine similarity using
    the pre-built FAISS index.

    Args:
        index_dir:      Directory containing index.faiss, metadata.json,
                        and mitigation_map.json (built by attack_embedding.py).
        top_k:          Return top-k technique matches. Default 1.
        min_confidence: Minimum cosine similarity to return a match. Default 0.3.
    """

    def __init__(
        self,
        index_dir: str = str(DEFAULT_INDEX_DIR),
        top_k: int = 1,
        min_confidence: float = 0.3,
    ):
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.min_confidence = min_confidence
        self._index = None
        self._metadata = []
        self._mitigation_map = {}
        self._encoder = None

    def _load(self):
        if self._index is not None:
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Install faiss and sentence-transformers:\n"
                "  pip install faiss-cpu sentence-transformers"
            )

        index_path = self.index_dir / "index.faiss"
        meta_path  = self.index_dir / "metadata.json"
        mit_path   = self.index_dir / "mitigation_map.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run 'python mitre_mapping/attack_embedding.py' first."
            )

        self._index = faiss.read_index(str(index_path))

        with meta_path.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if mit_path.exists():
            with mit_path.open("r", encoding="utf-8") as f:
                self._mitigation_map = json.load(f)

        self._encoder = SentenceTransformer(MODEL_NAME)
        logger.info("ATTACKTechniqueMapper loaded. Index has %d techniques.", self._index.ntotal)

    def map(self, description: str) -> list[TechniqueMatch]:
        """
        Map a vulnerability description to the top-k ATT&CK techniques.

        Args:
            description: Free-text vulnerability description.

        Returns:
            List of TechniqueMatch objects sorted by descending confidence.
        """
        self._load()

        vec = self._encoder.encode([description], normalize_embeddings=True).astype(np.float32)
        distances, indices = self._index.search(vec, self.top_k)

        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            confidence = float(dist)  # FAISS inner product = cosine sim for normalized vectors
            if confidence < self.min_confidence:
                continue

            meta = self._metadata[idx]
            tid = meta.get("technique_id", "T0000")
            mit = self._mitigation_map.get(tid, {})

            matches.append(TechniqueMatch(
                technique_id=tid,
                technique_name=meta.get("name", "Unknown"),
                tactic=meta.get("tactic", "Unknown"),
                confidence=confidence,
                mitigation_id=mit.get("mitigation_id"),
                mitigation_name=mit.get("mitigation_name"),
            ))

        return matches

    def map_entity(self, entity) -> Optional[TechniqueMatch]:
        """Convenience wrapper for a VulnerabilityEntity object."""
        results = self.map(entity.description)
        return results[0] if results else None


def main():
    import argparse
    from connectors.nvd_connector import NVDConnector

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Map a CVE to MITRE ATT&CK technique")
    parser.add_argument("--cve", help="CVE ID to look up and map, e.g. CVE-2024-12345")
    parser.add_argument("--description", help="Free-text description to map directly")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    mapper = ATTACKTechniqueMapper(index_dir=args.index_dir, top_k=args.top_k)

    if args.cve:
        connector = NVDConnector()
        entity = connector.fetch_by_cve_id(args.cve)
        if not entity:
            print(f"CVE not found: {args.cve}")
            return
        description = entity.description
        print(f"\nCVE: {args.cve}")
        print(f"CVSS: {entity.severity_score}  ({entity.severity_tier.value})")
        print(f"Description: {description[:200]}...")
    elif args.description:
        description = args.description
    else:
        parser.print_help()
        return

    matches = mapper.map(description)
    if not matches:
        print("\nNo ATT&CK technique matched above confidence threshold.")
        return

    print(f"\nTop-{args.top_k} ATT&CK Matches:")
    for i, m in enumerate(matches, 1):
        print(f"\n  [{i}] {m.technique_id}: {m.technique_name}")
        print(f"       Tactic:     {m.tactic}")
        print(f"       Confidence: {m.confidence:.3f}")
        if m.mitigation_id:
            print(f"       Mitigation: {m.mitigation_id} - {m.mitigation_name}")


if __name__ == "__main__":
    main()
