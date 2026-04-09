"""
connectors/stix_connector.py

Fetches STIX 2.1 bundles from a TAXII 2.1 server and normalizes
threat indicators, vulnerabilities, and threat actors to VulnerabilityEntity objects.

Default server: MITRE ATT&CK TAXII (cti-taxii.mitre.org)
OpenCTI community TAXII also supported.

Usage:
    python connectors/stix_connector.py --collection enterprise-attack --output data/stix_feed.jsonl

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from connectors.entity_schema import DataSource, VulnerabilityEntity

logger = logging.getLogger(__name__)

MITRE_TAXII_URL = "https://cti-taxii.mitre.org/taxii/"
MITRE_TAXII_COLLECTION_MAP = {
    "enterprise-attack": "95ecc380-afe9-11e4-9b6c-751b66dd541e",
    "mobile-attack":     "2f669986-b40b-4423-b720-4396ca6a462b",
    "ics-attack":        "02c3ef24-9cd4-48f3-a99f-b74ce24f1d34",
}

# Confidence normalization for STIX objects that carry a 0-100 confidence value
def _stix_confidence_to_cvss(confidence: Optional[int]) -> float:
    """Map STIX confidence (0-100) to CVSS-equivalent (0-10)."""
    if confidence is None:
        return 5.0
    return round(confidence / 10.0, 1)


class STIXConnector:
    """
    Reads STIX 2.1 bundles from a TAXII 2.1 server and emits VulnerabilityEntity objects.

    Handles three STIX object types:
        vulnerability:         direct CVE reference with CVSS score
        indicator:             threat actor indicator linked to a vulnerability
        threat-actor:          provides attribution context enriching co-occurrence features

    Args:
        taxii_url:      TAXII server root URL
        username:       TAXII authentication username (optional)
        password:       TAXII authentication password (optional)
    """

    def __init__(
        self,
        taxii_url: str = MITRE_TAXII_URL,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.taxii_url = taxii_url.rstrip("/")
        self.username = username or os.getenv("TAXII_USERNAME")
        self.password = password or os.getenv("TAXII_PASSWORD")

    def fetch_collection(self, collection_id: str) -> Generator[VulnerabilityEntity, None, None]:
        """Yield VulnerabilityEntity objects from a TAXII collection."""
        try:
            from taxii2client.v21 import Server
            import stix2
        except ImportError:
            raise ImportError("Install taxii2-client and stix2: pip install taxii2-client stix2")

        server = Server(
            self.taxii_url,
            user=self.username,
            password=self.password,
        )

        for api_root in server.api_roots:
            for collection in api_root.collections:
                if collection.id == collection_id:
                    bundle = collection.get_objects()
                    for obj in bundle.get("objects", []):
                        entity = self._parse_stix_object(obj)
                        if entity:
                            yield entity
                    return

        logger.warning("Collection not found: %s", collection_id)

    def fetch_from_bundle_file(self, bundle_path: str) -> Generator[VulnerabilityEntity, None, None]:
        """Parse a locally stored STIX 2.1 JSON bundle file."""
        import json
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

        with path.open("r", encoding="utf-8") as f:
            bundle = json.load(f)

        for obj in bundle.get("objects", []):
            entity = self._parse_stix_object(obj)
            if entity:
                yield entity

    def _parse_stix_object(self, obj: dict) -> Optional[VulnerabilityEntity]:
        obj_type = obj.get("type", "")

        if obj_type == "vulnerability":
            return self._parse_vulnerability(obj)
        if obj_type == "indicator":
            return self._parse_indicator(obj)
        return None

    def _parse_vulnerability(self, obj: dict) -> Optional[VulnerabilityEntity]:
        stix_id = obj.get("id", "")
        name = obj.get("name", "unknown")
        description = obj.get("description", name)

        # Extract CVE reference if present
        cve_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "cve":
                cve_id = ref.get("external_id", "")
                break

        score = _stix_confidence_to_cvss(obj.get("confidence"))

        # STIX vulnerability objects sometimes carry CVSS via extensions
        x_cvss = obj.get("x_cvss", {})
        if x_cvss.get("base_score"):
            try:
                score = float(x_cvss["base_score"])
            except (ValueError, TypeError):
                pass

        published_at = None
        created = obj.get("created")
        if created:
            try:
                published_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                pass

        return VulnerabilityEntity.create(
            source=DataSource.STIX_TAXII,
            source_id=cve_id or stix_id,
            severity_score=score,
            affected_component=name,
            description=description,
            published_at=published_at,
            raw=obj,
        )

    def _parse_indicator(self, obj: dict) -> Optional[VulnerabilityEntity]:
        stix_id = obj.get("id", "")
        description = obj.get("description", obj.get("name", "STIX indicator"))
        pattern = obj.get("pattern", "")
        confidence = obj.get("confidence")
        score = _stix_confidence_to_cvss(confidence)

        labels = obj.get("labels", [])
        component = labels[0] if labels else "unknown-indicator"

        published_at = None
        created = obj.get("created")
        if created:
            try:
                published_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                pass

        return VulnerabilityEntity.create(
            source=DataSource.STIX_TAXII,
            source_id=stix_id,
            severity_score=score,
            affected_component=component,
            description=f"{description} | Pattern: {pattern}",
            published_at=published_at,
            raw=obj,
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch STIX 2.1 threat intelligence from TAXII server")
    parser.add_argument("--collection", default="enterprise-attack", help="Collection name or UUID")
    parser.add_argument("--server", default=MITRE_TAXII_URL, help="TAXII server root URL")
    parser.add_argument("--bundle-file", default=None, help="Path to a local STIX 2.1 JSON bundle file")
    parser.add_argument("--output", default="data/stix_feed.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    connector = STIXConnector(taxii_url=args.server)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        if args.bundle_file:
            gen = connector.fetch_from_bundle_file(args.bundle_file)
        else:
            collection_id = MITRE_TAXII_COLLECTION_MAP.get(args.collection, args.collection)
            gen = connector.fetch_collection(collection_id)

        for entity in gen:
            f.write(entity.to_json() + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("Processed %d STIX objects so far", count)

    logger.info("Done. Wrote %d entities to %s", count, out_path)


if __name__ == "__main__":
    main()
