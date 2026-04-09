"""
connectors/code_repo_connector.py

Loads the CVEfixes dataset (Bhandari et al., PROMISE 2021) and normalizes
each vulnerable function record to a VulnerabilityEntity.

CVEfixes provides open-source commits that patch CVEs, covering C, C++,
Python, Java, and JavaScript. Each record links a vulnerable code function
to its CVE identifier and CVSS score.

Dataset source: https://github.com/secureIT-project/CVEfixes

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from connectors.entity_schema import DataSource, VulnerabilityEntity

logger = logging.getLogger(__name__)

CVEFIXES_DB_TABLE_METHODS = "method_change"
CVEFIXES_DB_TABLE_FILES   = "file_change"
CVEFIXES_DB_TABLE_COMMITS = "commits"
CVEFIXES_DB_TABLE_FIXES   = "fixes"
CVEFIXES_DB_TABLE_CVE     = "cve"


class CodeRepoConnector:
    """
    Loads vulnerability records from the CVEfixes SQLite database.

    The CVEfixes dataset organizes records across several tables.
    This connector joins them to extract:
        - CVE ID and CVSS score (from the cve table)
        - CWE classification (from the cve table)
        - Affected product and version (from the fixes/commits tables)
        - Vulnerable function source code (from method_change, as description)

    Args:
        db_path:   Path to the CVEfixes SQLite file (CVEfixes.db)
        language:  Optional filter by programming language ('Python', 'C', etc.)
    """

    def __init__(self, db_path: str, language: Optional[str] = None):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"CVEfixes database not found at {db_path}. "
                "Download it from https://github.com/secureIT-project/CVEfixes"
            )
        self.language = language

    def load(self) -> Generator[VulnerabilityEntity, None, None]:
        """Yield one VulnerabilityEntity per vulnerable method in CVEfixes."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield from self._query(conn)
        finally:
            conn.close()

    def _query(self, conn: sqlite3.Connection) -> Generator[VulnerabilityEntity, None, None]:
        lang_filter = ""
        params: list = []
        if self.language:
            lang_filter = "AND fc.programming_language = ?"
            params.append(self.language)

        sql = f"""
            SELECT
                cve.cve_id,
                cve.cvss3_base_score,
                cve.cwe_id,
                cve.description         AS cve_description,
                cve.published_date,
                commits.repo_url,
                file_change.programming_language,
                method_change.before_change AS vulnerable_code,
                method_change.name          AS method_name
            FROM method_change
            JOIN file_change fc          ON method_change.file_change_id = fc.file_change_id
            JOIN commits                 ON fc.hash = commits.hash
            JOIN fixes                   ON commits.hash = fixes.hash
            JOIN cve                     ON fixes.cve_id = cve.cve_id
            WHERE method_change.before_change IS NOT NULL
              AND method_change.before_change != ''
              {lang_filter}
            ORDER BY cve.published_date DESC
        """

        cursor = conn.execute(sql, params)
        seen_cve_method: set = set()

        for row in cursor:
            cve_id = row["cve_id"] or "UNKNOWN"
            method_name = row["method_name"] or "unknown_function"
            key = f"{cve_id}::{method_name}"
            if key in seen_cve_method:
                continue
            seen_cve_method.add(key)

            cvss_score = 0.0
            try:
                cvss_score = float(row["cvss3_base_score"] or 0)
            except (ValueError, TypeError):
                pass

            cwe_raw = row["cwe_id"] or ""
            cwe_ids = [c.strip() for c in cwe_raw.split(",") if c.strip().startswith("CWE-")]

            description = (
                f"Function: {method_name} | "
                f"Language: {row['programming_language'] or 'unknown'} | "
                f"CVE: {cve_id} | "
                f"{row['cve_description'] or ''}"
            )

            repo_url = row["repo_url"] or ""
            component = repo_url.split("/")[-1] if repo_url else "unknown-repo"

            published_at = None
            pub_str = row["published_date"]
            if pub_str:
                try:
                    from datetime import datetime, timezone
                    published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            yield VulnerabilityEntity.create(
                source=DataSource.CODE_REPO,
                source_id=cve_id,
                severity_score=cvss_score,
                affected_component=component,
                description=description,
                cwe_ids=cwe_ids,
                references=[repo_url] if repo_url else [],
                published_at=published_at,
                raw={
                    "method_name": method_name,
                    "language": row["programming_language"],
                    "vulnerable_code": (row["vulnerable_code"] or "")[:500],
                },
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Load all records into a pandas DataFrame for ML training."""
        records = []
        for entity in self.load():
            d = entity.to_dict()
            d["cwe_primary"] = entity.cwe_ids[0] if entity.cwe_ids else "UNKNOWN"
            records.append(d)
        return pd.DataFrame(records)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Load CVEfixes dataset into VulnerabilityEntity JSONL")
    parser.add_argument("--db", required=True, help="Path to CVEfixes.db SQLite file")
    parser.add_argument("--language", default=None, help="Filter by programming language")
    parser.add_argument("--output", default="data/cvefixes_feed.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    connector = CodeRepoConnector(db_path=args.db, language=args.language)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for entity in connector.load():
            f.write(entity.to_json() + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("Processed %d method records so far", count)

    logger.info("Done. Wrote %d entities to %s", count, out_path)


if __name__ == "__main__":
    main()
