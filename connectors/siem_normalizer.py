"""
connectors/siem_normalizer.py

Parses Syslog (RFC 5424) and CEF (Common Event Format) log lines
and normalizes security-relevant events to VulnerabilityEntity objects.

A rule-based severity classifier assigns a CVSS-equivalent score
based on event taxonomy, facility, and keyword signals in the message field.

Usage:
    python connectors/siem_normalizer.py --input data/siem.log --format syslog --output data/siem_feed.jsonl

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from connectors.entity_schema import DataSource, VulnerabilityEntity

logger = logging.getLogger(__name__)

# Syslog severity levels map to CVSS-equivalent scores
SYSLOG_SEVERITY_TO_SCORE = {
    0: 10.0,  # Emergency
    1: 9.5,   # Alert
    2: 9.0,   # Critical
    3: 7.5,   # Error
    4: 5.0,   # Warning
    5: 3.0,   # Notice
    6: 1.5,   # Informational
    7: 0.5,   # Debug
}

# CEF severity 0-10 maps directly to CVSS
def cef_severity_to_score(cef_sev: int) -> float:
    return max(0.0, min(10.0, float(cef_sev)))

# Keywords that signal security relevance; higher score for higher-risk keywords
SECURITY_KEYWORDS = {
    "exploit":           9.5,
    "injection":         9.0,
    "privilege_escalation": 9.0,
    "buffer overflow":   8.5,
    "remote code":       9.5,
    "unauthorized":      7.0,
    "brute force":       7.5,
    "credential":        6.5,
    "malware":           8.0,
    "ransomware":        9.5,
    "phishing":          6.0,
    "xss":               6.5,
    "sql injection":     8.5,
    "command injection": 9.0,
    "zero-day":          10.0,
    "zero day":          10.0,
    "denial of service": 7.5,
    "dos":               7.0,
    "portscan":          4.0,
    "authentication failure": 5.5,
    "login failed":      4.5,
    "permission denied": 3.5,
}

# RFC 5424 syslog pattern
SYSLOG_PATTERN = re.compile(
    r"<(?P<priority>\d+)>"
    r"(?P<version>\d+)?\s*"
    r"(?P<timestamp>\S+)\s+"
    r"(?P<hostname>\S+)\s+"
    r"(?P<appname>\S+)\s+"
    r"(?P<procid>\S+)\s+"
    r"(?P<msgid>\S+)\s+"
    r"(?P<structured_data>\[.*?\]|-)\s*"
    r"(?P<message>.*)"
)

# CEF format pattern
CEF_PATTERN = re.compile(
    r"CEF:(?P<version>\d+)\|"
    r"(?P<device_vendor>[^|]*)\|"
    r"(?P<device_product>[^|]*)\|"
    r"(?P<device_version>[^|]*)\|"
    r"(?P<signature_id>[^|]*)\|"
    r"(?P<name>[^|]*)\|"
    r"(?P<severity>[^|]*)\|"
    r"(?P<extension>.*)"
)


def classify_security_score(message: str, base_score: float) -> float:
    """
    Boost or confirm the base score using keyword signals in the log message.
    Returns the higher of the base score and any keyword match.
    """
    lower = message.lower()
    keyword_score = base_score
    for keyword, score in SECURITY_KEYWORDS.items():
        if keyword in lower:
            keyword_score = max(keyword_score, score)
    return keyword_score


def parse_syslog_line(line: str) -> Optional[VulnerabilityEntity]:
    m = SYSLOG_PATTERN.match(line.strip())
    if not m:
        return None

    priority = int(m.group("priority"))
    severity_code = priority % 8
    base_score = SYSLOG_SEVERITY_TO_SCORE.get(severity_code, 1.0)

    hostname = m.group("hostname") or "unknown-host"
    appname = m.group("appname") or "unknown-app"
    message = m.group("message") or ""

    score = classify_security_score(message, base_score)

    # Only forward events above low-severity threshold
    if score < 3.0:
        return None

    timestamp_str = m.group("timestamp")
    published_at = None
    try:
        published_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        published_at = datetime.now(timezone.utc)

    return VulnerabilityEntity.create(
        source=DataSource.SIEM_LOG,
        source_id=f"syslog-{hostname}-{timestamp_str}",
        severity_score=score,
        affected_component=f"{hostname}/{appname}",
        description=message,
        published_at=published_at,
        raw={"line": line, "facility": priority // 8, "severity": severity_code},
    )


def parse_cef_line(line: str) -> Optional[VulnerabilityEntity]:
    m = CEF_PATTERN.match(line.strip())
    if not m:
        return None

    vendor = m.group("device_vendor") or "unknown-vendor"
    product = m.group("device_product") or "unknown-product"
    name = m.group("name") or ""
    ext = m.group("extension") or ""

    try:
        sev = int(m.group("severity"))
    except (ValueError, TypeError):
        sev = 5
    base_score = cef_severity_to_score(sev)
    score = classify_security_score(f"{name} {ext}", base_score)

    if score < 3.0:
        return None

    # Parse extension key=value pairs
    ext_dict: dict = {}
    for pair in re.findall(r"(\w+)=([^\s]+(?:\s[^\w=].*?)?)", ext):
        ext_dict[pair[0]] = pair[1]

    published_at = datetime.now(timezone.utc)
    if "rt" in ext_dict:
        try:
            ms = int(ext_dict["rt"])
            published_at = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        except (ValueError, TypeError):
            pass

    return VulnerabilityEntity.create(
        source=DataSource.SIEM_LOG,
        source_id=f"cef-{m.group('signature_id')}-{ext_dict.get('src', 'unknown')}",
        severity_score=score,
        affected_component=f"{vendor}/{product}",
        description=f"{name} | {ext}",
        published_at=published_at,
        raw={"line": line, "extension": ext_dict},
    )


class SIEMNormalizer:
    """
    Reads a log file (syslog or CEF format) line by line and yields VulnerabilityEntity objects.

    Args:
        log_format: 'syslog' (RFC 5424) or 'cef' (Common Event Format)
    """

    def __init__(self, log_format: str = "syslog"):
        if log_format not in ("syslog", "cef"):
            raise ValueError(f"Unsupported log format: {log_format}. Use 'syslog' or 'cef'.")
        self.log_format = log_format
        self._parser = parse_syslog_line if log_format == "syslog" else parse_cef_line

    def process_file(self, path: str) -> Generator[VulnerabilityEntity, None, None]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                entity = self._parser(line)
                if entity:
                    yield entity

    def process_lines(self, lines: list[str]) -> Generator[VulnerabilityEntity, None, None]:
        for line in lines:
            entity = self._parser(line)
            if entity:
                yield entity


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Normalize SIEM log files to VulnerabilityEntity JSONL")
    parser.add_argument("--input", required=True, help="Path to input log file")
    parser.add_argument("--format", choices=["syslog", "cef"], default="syslog", help="Log format")
    parser.add_argument("--output", default="data/siem_feed.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    normalizer = SIEMNormalizer(log_format=args.format)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for entity in normalizer.process_file(args.input):
            f.write(entity.to_json() + "\n")
            count += 1

    logger.info("Done. Wrote %d security events (%d skipped below threshold) to %s", count, skipped, out_path)


if __name__ == "__main__":
    main()
