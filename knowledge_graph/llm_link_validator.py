"""
knowledge_graph/llm_link_validator.py

GPT-4 candidate link generator and BERT semantic similarity validator.

This module implements the LLM-augmented link prediction step in Layer 2.
It takes candidate (subject, predicate, object) triples discovered by the GAT,
asks GPT-4 to justify whether the link is semantically valid, then scores
the justification against existing node descriptions using BERT cosine similarity.

Only candidate links that pass the similarity gate (default threshold 0.6)
are added to the knowledge graph.

This is an offline batch process. It does not run at inference time.
Online inference uses only the pre-built graph and the XGBoost ensemble.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SIMILARITY_GATE  = 0.60   # Minimum BERT cosine similarity to accept a link
MAX_RETRIES      = 3
RETRY_SLEEP_SECS = 5


@dataclass
class ValidatedLink:
    src_description:  str
    dst_description:  str
    predicate:        str
    justification:    str   # GPT-4 generated explanation
    similarity_score: float  # BERT cosine similarity of justification to node descriptions
    accepted:         bool   # Whether similarity_score >= SIMILARITY_GATE


class LLMLinkValidator:
    """
    Validates candidate knowledge graph links using GPT-4 and BERT similarity.

    Validation pipeline per candidate:
        1. Compose a prompt from the source and target node descriptions.
        2. Call GPT-4 to generate a justification for or against the link.
        3. Encode the justification and both node descriptions with BERT.
        4. Compute cosine similarity between the justification and the mean
           of the two node description embeddings.
        5. Accept the link if similarity >= SIMILARITY_GATE.

    Args:
        openai_api_key:     OpenAI API key. Falls back to OPENAI_API_KEY env var.
        similarity_gate:    Minimum cosine similarity to accept a candidate link.
        bert_model:         Sentence-transformer model for similarity scoring.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        similarity_gate: float = SIMILARITY_GATE,
        bert_model: str = "all-MiniLM-L6-v2",
    ):
        self.api_key         = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.similarity_gate = similarity_gate
        self.bert_model_name = bert_model
        self._encoder        = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.bert_model_name)
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        return self._encoder

    def _cosine_similarity(self, a, b) -> float:
        import numpy as np
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))

    def _call_gpt4(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.30.0")

        if not self.api_key:
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")

        client = OpenAI(api_key=self.api_key)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("GPT-4 call failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_SLEEP_SECS * (attempt + 1))

        return ""

    def _build_prompt(self, src_desc: str, predicate: str, dst_desc: str) -> str:
        return (
            f"You are a cybersecurity knowledge graph expert. "
            f"A vulnerability knowledge graph has two nodes:\n\n"
            f"Node A: {src_desc[:300]}\n\n"
            f"Node B: {dst_desc[:300]}\n\n"
            f"Should Node A have a '{predicate}' relationship to Node B? "
            f"Briefly justify your answer in one to two sentences. "
            f"Focus on technical cybersecurity reasoning."
        )

    def validate(
        self,
        src_description: str,
        dst_description: str,
        predicate: str,
    ) -> ValidatedLink:
        """
        Validate a single candidate link using GPT-4 + BERT similarity.

        Args:
            src_description:    Free-text description of the source node.
            dst_description:    Free-text description of the target node.
            predicate:          Proposed relationship type ('affects', 'exploitedBy', etc.).

        Returns:
            ValidatedLink with acceptance decision and similarity score.
        """
        prompt = self._build_prompt(src_description, predicate, dst_description)
        justification = self._call_gpt4(prompt)

        if not justification:
            return ValidatedLink(
                src_description=src_description,
                dst_description=dst_description,
                predicate=predicate,
                justification="",
                similarity_score=0.0,
                accepted=False,
            )

        encoder = self._get_encoder()
        import numpy as np
        vecs = encoder.encode(
            [justification, src_description[:512], dst_description[:512]],
            normalize_embeddings=True,
        )
        just_vec = vecs[0]
        node_mean = (vecs[1] + vecs[2]) / 2.0
        sim = self._cosine_similarity(just_vec, node_mean)

        accepted = sim >= self.similarity_gate

        if accepted:
            logger.info(
                "Link ACCEPTED: '%s' -[%s]-> '%s' (sim=%.3f)",
                src_description[:60], predicate, dst_description[:60], sim,
            )
        else:
            logger.debug(
                "Link REJECTED: sim=%.3f < %.3f", sim, self.similarity_gate,
            )

        return ValidatedLink(
            src_description=src_description,
            dst_description=dst_description,
            predicate=predicate,
            justification=justification,
            similarity_score=sim,
            accepted=accepted,
        )

    def validate_batch(
        self,
        candidates: list[dict],
        rate_limit_sleep: float = 1.0,
    ) -> list[ValidatedLink]:
        """
        Validate a batch of candidate links.

        Args:
            candidates:         List of dicts with keys: src_description, dst_description, predicate.
            rate_limit_sleep:   Seconds to sleep between GPT-4 calls.

        Returns:
            List of ValidatedLink objects.
        """
        results = []
        for i, c in enumerate(candidates):
            result = self.validate(
                src_description=c["src_description"],
                dst_description=c["dst_description"],
                predicate=c.get("predicate", "coOccursWith"),
            )
            results.append(result)
            if i < len(candidates) - 1:
                time.sleep(rate_limit_sleep)

        accepted_count = sum(1 for r in results if r.accepted)
        logger.info(
            "Batch validation complete. %d/%d candidates accepted.",
            accepted_count, len(candidates),
        )
        return results
