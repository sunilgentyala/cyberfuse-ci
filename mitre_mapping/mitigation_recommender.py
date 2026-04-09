"""
mitre_mapping/mitigation_recommender.py

Recommends MITRE ATT&CK mitigation controls for a detected vulnerability
based on the technique match returned by technique_mapper.py.

The recommender reads from the mitigation_map.json file built by
attack_embedding.py and also pulls full mitigation descriptions from the
ATT&CK STIX bundle for richer output.

Output per recommendation:
    mitigation_id:          ATT&CK mitigation ID (M-series)
    mitigation_name:        Human-readable name
    mitigation_description: Full description from ATT&CK
    technique_id:           The technique this mitigation addresses
    technique_name:         Human-readable technique name
    confidence:             Inherited from technique mapper cosine similarity

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mitre_mapping.technique_mapper import ATTACKTechniqueMapper, TechniqueMatch

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).parent / "index"


@dataclass
class MitigationRecommendation:
    mitigation_id:          str
    mitigation_name:        str
    mitigation_description: str
    technique_id:           str
    technique_name:         str
    tactic:                 str
    confidence:             float

    def to_dict(self) -> dict:
        return {
            "mitigation_id":          self.mitigation_id,
            "mitigation_name":        self.mitigation_name,
            "mitigation_description": self.mitigation_description,
            "technique_id":           self.technique_id,
            "technique_name":         self.technique_name,
            "tactic":                 self.tactic,
            "confidence":             round(self.confidence, 4),
        }

    def to_report_line(self) -> str:
        return (
            f"Technique : {self.technique_id} - {self.technique_name} ({self.tactic})\n"
            f"Confidence: {self.confidence:.3f}\n"
            f"Mitigation: {self.mitigation_id} - {self.mitigation_name}\n"
            f"Detail    : {self.mitigation_description[:300]}..."
        )


class MitigationRecommender:
    """
    Wraps ATTACKTechniqueMapper to produce full mitigation recommendations.

    The ATT&CK mitigation catalog contains M-series controls that map to
    one or more techniques. This recommender resolves the full mitigation
    description from the index files and returns structured recommendations
    ready for consumption by ticketing and patch management systems.

    Args:
        index_dir:      Directory containing index files from attack_embedding.py.
        top_k:          Number of technique matches to consider per vulnerability.
    """

    def __init__(
        self,
        index_dir: str = str(DEFAULT_INDEX_DIR),
        top_k: int = 3,
    ):
        self.mapper    = ATTACKTechniqueMapper(index_dir=index_dir, top_k=top_k)
        self.index_dir = Path(index_dir)
        self._mit_details: dict = {}

    def _load_mitigation_details(self):
        if self._mit_details:
            return
        path = self.index_dir / "mitigation_map.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # raw is {technique_id: {mitigation_id, mitigation_name}}
            # Build reverse: mitigation_id -> description (name used as description fallback)
            for tid, data in raw.items():
                mid = data.get("mitigation_id", "")
                if mid and mid not in self._mit_details:
                    self._mit_details[mid] = data.get("mitigation_name", "See ATT&CK catalog")
        else:
            logger.warning("mitigation_map.json not found at %s. Run attack_embedding.py first.", self.index_dir)

    def recommend(self, description: str) -> list[MitigationRecommendation]:
        """
        Generate mitigation recommendations for a vulnerability description.

        Args:
            description:    Free-text vulnerability description.

        Returns:
            List of MitigationRecommendation objects, one per technique match
            that has a mapped mitigation. Sorted by descending confidence.
        """
        self._load_mitigation_details()
        matches: list[TechniqueMatch] = self.mapper.map(description)

        recommendations = []
        for match in matches:
            if not match.mitigation_id:
                continue

            mit_desc = self._mit_details.get(match.mitigation_id, match.mitigation_name or "")

            recommendations.append(MitigationRecommendation(
                mitigation_id=match.mitigation_id,
                mitigation_name=match.mitigation_name or "",
                mitigation_description=mit_desc,
                technique_id=match.technique_id,
                technique_name=match.technique_name,
                tactic=match.tactic,
                confidence=match.confidence,
            ))

        return recommendations

    def recommend_for_entity(self, entity) -> list[MitigationRecommendation]:
        """Convenience wrapper for a VulnerabilityEntity object."""
        return self.recommend(entity.description)
