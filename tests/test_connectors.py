"""
tests/test_connectors.py

Unit tests for Layer 1 connectors.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

import json
import pytest
from datetime import datetime, timezone

from connectors.entity_schema import (
    VulnerabilityEntity,
    DataSource,
    SeverityTier,
    cvss_to_tier,
)
from connectors.siem_normalizer import (
    SIEMNormalizer,
    parse_syslog_line,
    parse_cef_line,
    classify_security_score,
)


class TestEntitySchema:
    def test_create_basic(self):
        entity = VulnerabilityEntity.create(
            source=DataSource.CVE_NVD,
            source_id="CVE-2024-99999",
            severity_score=9.8,
            affected_component="openssl",
            description="Critical heap overflow in OpenSSL.",
        )
        assert entity.source == DataSource.CVE_NVD
        assert entity.source_id == "CVE-2024-99999"
        assert entity.severity_score == 9.8
        assert entity.severity_tier == SeverityTier.CRITICAL
        assert entity.affected_component == "openssl"

    def test_score_clamped_to_range(self):
        entity = VulnerabilityEntity.create(
            source=DataSource.CVE_NVD,
            source_id="test",
            severity_score=15.0,
            affected_component="test",
            description="test",
        )
        assert entity.severity_score == 10.0

        entity_neg = VulnerabilityEntity.create(
            source=DataSource.CVE_NVD,
            source_id="test",
            severity_score=-1.0,
            affected_component="test",
            description="test",
        )
        assert entity_neg.severity_score == 0.0

    def test_to_json_roundtrip(self):
        entity = VulnerabilityEntity.create(
            source=DataSource.STIX_TAXII,
            source_id="indicator--abc123",
            severity_score=7.5,
            affected_component="apache",
            description="Threat indicator for Apache exploit.",
            cwe_ids=["CWE-79"],
            references=["https://example.com"],
            published_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        j = entity.to_json()
        d = json.loads(j)
        assert d["source"] == "stix_taxii"
        assert d["severity_tier"] == "HIGH"
        assert d["cwe_ids"] == ["CWE-79"]

    @pytest.mark.parametrize("score,expected_tier", [
        (9.5,  SeverityTier.CRITICAL),
        (7.0,  SeverityTier.HIGH),
        (5.5,  SeverityTier.MEDIUM),
        (2.0,  SeverityTier.LOW),
        (0.0,  SeverityTier.INFORMATIONAL),
    ])
    def test_cvss_to_tier(self, score, expected_tier):
        assert cvss_to_tier(score) == expected_tier


class TestSIEMNormalizer:
    def test_parse_syslog_high_severity(self):
        # Priority 8 = facility 1 (user), severity 0 (Emergency) -> score 10.0
        line = "<8>1 2025-06-01T12:00:00Z webserver.internal nginx 1234 - - SQL injection detected in POST /login"
        entity = parse_syslog_line(line)
        assert entity is not None
        assert entity.source.value == "siem_log"
        assert entity.severity_score >= 8.0

    def test_parse_syslog_low_severity_filtered(self):
        # Priority 191 = facility 23, severity 7 (Debug) with no keywords -> score 0.5, filtered out
        line = "<191>1 2025-06-01T12:00:00Z host app 1 - - routine health check ok"
        entity = parse_syslog_line(line)
        assert entity is None

    def test_parse_cef_high_severity(self):
        line = "CEF:0|Palo Alto|Firewall|10.0|threat-1234|Remote code execution detected|9|src=192.168.1.1 dst=10.0.0.5 rt=1748736000000"
        entity = parse_cef_line(line)
        assert entity is not None
        assert entity.severity_score >= 9.0

    def test_siem_normalizer_file_not_found(self):
        normalizer = SIEMNormalizer(log_format="syslog")
        with pytest.raises(FileNotFoundError):
            list(normalizer.process_file("/nonexistent/path/to/siem.log"))

    def test_keyword_score_boost(self):
        score = classify_security_score("SQL injection attack detected in web application", 2.0)
        assert score >= 8.0  # keyword 'sql injection' should boost to 8.5

    def test_siem_normalizer_process_lines(self):
        normalizer = SIEMNormalizer(log_format="syslog")
        lines = [
            "<8>1 2025-06-01T12:00:00Z host app 1 - - exploit detected in buffer overflow attempt",
            "<191>1 2025-06-01T12:01:00Z host app 2 - - routine debug message",
        ]
        results = list(normalizer.process_lines(lines))
        assert len(results) == 1
        assert results[0].severity_score >= 9.0
