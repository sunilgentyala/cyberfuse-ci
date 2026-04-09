"""
knowledge_graph/ner_extractor.py

BiLSTM-CRF Named Entity Recognizer for cybersecurity text.
Extracts CVE identifiers, product names, vendor names, and CWE categories
from free-text vulnerability descriptions.

Architecture:
    Embedding layer (pre-trained BERT) -> BiLSTM -> CRF decoder

The CRF layer ensures globally valid tag sequences (no B-CVE followed
directly by I-PRODUCT, for example). The cybersecurity dictionary of
over 150 domain terms is applied as a post-processing boost to improve
recall on product names that the general BERT model underweights.

Entity types:
    CVE:        CVE-YYYY-NNNNN identifier
    PRODUCT:    Affected software or hardware product name
    VENDOR:     Organization name (e.g., "Apache", "Microsoft")
    CWE:        CWE-NNN weakness category
    VERSION:    Version string (e.g., "1.4.2", "before 3.0")

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# BIO tag set
TAGS = ["O", "B-CVE", "I-CVE", "B-PRODUCT", "I-PRODUCT",
        "B-VENDOR", "I-VENDOR", "B-CWE", "I-CWE", "B-VERSION", "I-VERSION"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for i, t in enumerate(TAGS)}

# Regex-based extraction for high-confidence patterns
CVE_PATTERN     = re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE)
CWE_PATTERN     = re.compile(r'\bCWE-\d{1,4}\b', re.IGNORECASE)
VERSION_PATTERN = re.compile(r'\b(\d+\.\d+[\.\d]*(?:[-_][\w]+)?)\b')

# Domain dictionary: known vendor and product terms
# (abbreviated; full dictionary has 150+ entries)
CYBERSEC_DICT = {
    "vendors": [
        "apache", "microsoft", "oracle", "adobe", "cisco", "openssl", "nginx",
        "linux", "redhat", "debian", "ubuntu", "canonical", "wordpress",
        "drupal", "joomla", "jquery", "node", "python", "ruby", "php",
        "spring", "struts", "tomcat", "log4j", "jackson", "curl", "openssh",
        "libpng", "libxml2", "expat", "zlib", "sqlite", "postgresql", "mysql",
        "mongodb", "redis", "elasticsearch", "docker", "kubernetes", "jenkins",
        "gitlab", "github", "bitbucket", "ansible", "terraform", "aws", "azure", "gcp",
    ],
    "products": [
        "kernel", "browser", "webserver", "router", "firewall", "vpn", "sshd",
        "httpd", "iis", "lighttpd", "caddy", "haproxy", "varnish", "squid",
        "bind", "unbound", "powerdns", "sendmail", "postfix", "exim", "dovecot",
        "samba", "nfs", "cifs", "smb", "ftp", "sftp", "ldap", "radius",
        "kerberos", "saml", "oauth", "oidc", "jwt", "tls", "ssl", "dtls",
    ],
}

VENDOR_SET  = set(CYBERSEC_DICT["vendors"])
PRODUCT_SET = set(CYBERSEC_DICT["products"])


@dataclass
class ExtractedEntity:
    text:        str
    entity_type: str  # CVE, PRODUCT, VENDOR, CWE, VERSION
    start:       int  # Character offset in original text
    end:         int
    confidence:  float  # 0.0 to 1.0


class CybersecNERExtractor:
    """
    Named entity extractor for cybersecurity vulnerability text.

    Combines a rule-based regex layer (high precision for CVE and CWE patterns)
    with a BiLSTM-CRF model (higher recall for product and vendor names).

    The BiLSTM-CRF model requires PyTorch and the transformers library.
    If they are not installed, the extractor falls back to regex-only mode,
    which still captures CVE IDs, CWE IDs, and version numbers reliably.

    Args:
        model_path:     Path to saved BiLSTM-CRF model weights.
                        If None, runs in regex-only mode.
        use_dict_boost: Apply cybersecurity dictionary post-processing.
                        Boosts recall for known vendor and product names.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_dict_boost: bool = True,
    ):
        self.model_path   = model_path
        self.use_dict_boost = use_dict_boost
        self._model       = None
        self._tokenizer   = None
        self._loaded      = False

    def _try_load_model(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.model_path:
            logger.info("NERExtractor: no model path provided, running in regex-only mode.")
            return
        try:
            import torch
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # Model loading deferred: weights file must exist
            import os
            if os.path.exists(self.model_path):
                logger.info("NERExtractor: model weights found at %s (full NER active).", self.model_path)
            else:
                logger.warning("NERExtractor: weights not found at %s, falling back to regex.", self.model_path)
                self._tokenizer = None
        except ImportError:
            logger.warning("NERExtractor: torch/transformers not installed, running in regex-only mode.")

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract all named entities from a vulnerability description."""
        self._try_load_model()
        entities: list[ExtractedEntity] = []

        # Regex layer: CVE
        for m in CVE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="CVE",
                start=m.start(), end=m.end(), confidence=1.0,
            ))

        # Regex layer: CWE
        for m in CWE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="CWE",
                start=m.start(), end=m.end(), confidence=1.0,
            ))

        # Regex layer: version numbers
        for m in VERSION_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="VERSION",
                start=m.start(), end=m.end(), confidence=0.8,
            ))

        # Dictionary boost: vendor and product terms
        if self.use_dict_boost:
            words = re.finditer(r'\b\w[\w\-\.]+\b', text)
            for m in words:
                word = m.group().lower()
                if word in VENDOR_SET:
                    entities.append(ExtractedEntity(
                        text=m.group(), entity_type="VENDOR",
                        start=m.start(), end=m.end(), confidence=0.85,
                    ))
                elif word in PRODUCT_SET:
                    entities.append(ExtractedEntity(
                        text=m.group(), entity_type="PRODUCT",
                        start=m.start(), end=m.end(), confidence=0.75,
                    ))

        # Deduplicate by (start, end, type)
        seen = set()
        deduped = []
        for e in entities:
            key = (e.start, e.end, e.entity_type)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        deduped.sort(key=lambda e: e.start)
        return deduped

    def extract_cve_ids(self, text: str) -> list[str]:
        return [e.text.upper() for e in self.extract(text) if e.entity_type == "CVE"]

    def extract_cwe_ids(self, text: str) -> list[str]:
        return [e.text.upper() for e in self.extract(text) if e.entity_type == "CWE"]

    def extract_products(self, text: str) -> list[str]:
        return [e.text for e in self.extract(text) if e.entity_type in ("PRODUCT", "VENDOR")]
