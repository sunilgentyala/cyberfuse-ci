"""
connectors/entity_schema.py

Defines the shared VulnerabilityEntity tuple that all Layer 1 connectors
normalize their source data into. Every connector in this package produces
VulnerabilityEntity objects. The knowledge graph builder in Layer 2 consumes them.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import json
import uuid


class DataSource(str, Enum):
    CVE_NVD      = "cve_nvd"
    STIX_TAXII   = "stix_taxii"
    SIEM_LOG     = "siem_log"
    CODE_REPO    = "code_repo"


class SeverityTier(str, Enum):
    CRITICAL      = "CRITICAL"
    HIGH          = "HIGH"
    MEDIUM        = "MEDIUM"
    LOW           = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


def cvss_to_tier(score: float) -> SeverityTier:
    """Map a CVSS v3.1 base score to a SeverityTier.

    Thresholds follow NVD conventions:
        9.0-10.0  Critical
        7.0-8.9   High
        4.0-6.9   Medium
        0.1-3.9   Low
        0.0       Informational
    """
    if score >= 9.0:
        return SeverityTier.CRITICAL
    if score >= 7.0:
        return SeverityTier.HIGH
    if score >= 4.0:
        return SeverityTier.MEDIUM
    if score > 0.0:
        return SeverityTier.LOW
    return SeverityTier.INFORMATIONAL


@dataclass
class VulnerabilityEntity:
    """
    Canonical intermediate representation produced by every Layer 1 connector.

    All source-specific fields (CVE ID, STIX object ID, SIEM event ID, commit
    hash) map into this schema before reaching the knowledge graph builder.
    """

    entity_id:           str              # Framework-internal UUID
    source:              DataSource       # Which connector produced this entity
    source_id:           str              # Native ID (CVE-2024-12345, STIX UUID, etc.)
    severity_score:      float            # CVSS-normalised 0.0 to 10.0
    severity_tier:       SeverityTier     # Derived from severity_score
    affected_component:  str              # Product or software component name
    affected_versions:   list[str]        # Version strings or ranges
    description:         str              # Free-text description for NER
    cwe_ids:             list[str]        # CWE identifiers e.g. ['CWE-79', 'CWE-89']
    references:          list[str]        # External URLs
    published_at:        Optional[datetime]
    modified_at:         Optional[datetime]
    raw:                 dict = field(default_factory=dict, repr=False)  # Original record

    @classmethod
    def create(
        cls,
        source: DataSource,
        source_id: str,
        severity_score: float,
        affected_component: str,
        description: str,
        affected_versions: Optional[list[str]] = None,
        cwe_ids: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
        published_at: Optional[datetime] = None,
        modified_at: Optional[datetime] = None,
        raw: Optional[dict] = None,
    ) -> "VulnerabilityEntity":
        score = max(0.0, min(10.0, severity_score))
        return cls(
            entity_id=str(uuid.uuid4()),
            source=source,
            source_id=source_id,
            severity_score=score,
            severity_tier=cvss_to_tier(score),
            affected_component=affected_component,
            affected_versions=affected_versions or [],
            description=description,
            cwe_ids=cwe_ids or [],
            references=references or [],
            published_at=published_at,
            modified_at=modified_at,
            raw=raw or {},
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["severity_tier"] = self.severity_tier.value
        d["published_at"] = self.published_at.isoformat() if self.published_at else None
        d["modified_at"] = self.modified_at.isoformat() if self.modified_at else None
        d.pop("raw")
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
