"""
connectors/nvd_connector.py

Ingests CVE records from the NIST NVD REST API v2.0 and normalizes them
to VulnerabilityEntity objects.

Usage:
    python connectors/nvd_connector.py --days 7 --output data/cve_feed.jsonl

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

import requests

from connectors.entity_schema import DataSource, VulnerabilityEntity

logger = logging.getLogger(__name__)

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_PAGE_SIZE = 2000
RATE_LIMIT_SLEEP = 6.0  # seconds between requests without an API key


class NVDConnector:
    """
    Fetches CVE records from the NVD API and emits VulnerabilityEntity objects.

    The NVD API returns paginated results. This connector handles pagination
    automatically and respects the rate limit (5 requests per 30 seconds without
    an API key, or 50 per 30 seconds with one).

    Args:
        api_key:    NVD API key. Obtain from https://nvd.nist.gov/developers/request-an-api-key
                    Optional but strongly recommended to avoid rate limiting.
        page_size:  Number of CVEs to fetch per API call. Max 2000.
    """

    def __init__(self, api_key: Optional[str] = None, page_size: int = DEFAULT_PAGE_SIZE):
        self.api_key = api_key or os.getenv("NVD_API_KEY")
        self.page_size = min(page_size, DEFAULT_PAGE_SIZE)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["apiKey"] = self.api_key

    def fetch_recent(self, days: int = 7) -> Generator[VulnerabilityEntity, None, None]:
        """Yield all CVEs published or modified in the last N days."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        yield from self._paginate(
            pub_start=start.strftime("%Y-%m-%dT%H:%M:%S.000"),
            pub_end=end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        )

    def fetch_by_cve_id(self, cve_id: str) -> Optional[VulnerabilityEntity]:
        """Fetch a single CVE by its ID, e.g. CVE-2024-12345."""
        params = {"cveId": cve_id}
        data = self._get(params)
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            logger.warning("CVE not found: %s", cve_id)
            return None
        return self._parse_cve(vulns[0].get("cve", {}))

    def _paginate(self, **params) -> Generator[VulnerabilityEntity, None, None]:
        start_index = 0
        total_results = None

        while True:
            page_params = {**params, "startIndex": start_index, "resultsPerPage": self.page_size}
            data = self._get(page_params)

            if total_results is None:
                total_results = data.get("totalResults", 0)
                logger.info("Total CVEs to fetch: %d", total_results)

            for item in data.get("vulnerabilities", []):
                cve = item.get("cve", {})
                entity = self._parse_cve(cve)
                if entity:
                    yield entity

            start_index += self.page_size
            if start_index >= (total_results or 0):
                break

            sleep_s = RATE_LIMIT_SLEEP if not self.api_key else 0.6
            time.sleep(sleep_s)

    def _get(self, params: dict) -> dict:
        try:
            resp = self.session.get(NVD_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("NVD API request failed: %s", exc)
            return {}

    def _parse_cve(self, cve: dict) -> Optional[VulnerabilityEntity]:
        cve_id = cve.get("id", "")
        if not cve_id:
            return None

        description = ""
        for d in cve.get("descriptions", []):
            if d.get("lang") == "en":
                description = d.get("value", "")
                break

        cvss_score = 0.0
        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            entries = metrics.get(key, [])
            if entries:
                cvss_score = entries[0].get("cvssData", {}).get("baseScore", 0.0)
                break

        cwe_ids = []
        for weakness in cve.get("weaknesses", []):
            for wd in weakness.get("description", []):
                val = wd.get("value", "")
                if val.startswith("CWE-"):
                    cwe_ids.append(val)

        affected_components = []
        references = [r.get("url", "") for r in cve.get("references", [])]

        for config in cve.get("configurations", []):
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    cpe = match.get("criteria", "")
                    parts = cpe.split(":")
                    if len(parts) > 4:
                        affected_components.append(parts[4])

        component = affected_components[0] if affected_components else "unknown"

        published_at = None
        modified_at = None
        try:
            pub = cve.get("published")
            if pub:
                published_at = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            mod = cve.get("lastModified")
            if mod:
                modified_at = datetime.fromisoformat(mod.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        return VulnerabilityEntity.create(
            source=DataSource.CVE_NVD,
            source_id=cve_id,
            severity_score=cvss_score,
            affected_component=component,
            description=description,
            affected_versions=[],
            cwe_ids=list(set(cwe_ids)),
            references=[r for r in references if r],
            published_at=published_at,
            modified_at=modified_at,
            raw=cve,
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest recent CVEs from NVD API v2.0")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--output", type=str, default="data/cve_feed.jsonl", help="Output JSONL file path")
    parser.add_argument("--cve-id", type=str, default=None, help="Fetch a single CVE by ID")
    args = parser.parse_args()

    connector = NVDConnector()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        if args.cve_id:
            entity = connector.fetch_by_cve_id(args.cve_id)
            if entity:
                f.write(entity.to_json() + "\n")
                count = 1
        else:
            for entity in connector.fetch_recent(days=args.days):
                f.write(entity.to_json() + "\n")
                count += 1
                if count % 100 == 0:
                    logger.info("Fetched %d CVEs so far", count)

    logger.info("Done. Wrote %d entities to %s", count, out_path)


if __name__ == "__main__":
    main()
