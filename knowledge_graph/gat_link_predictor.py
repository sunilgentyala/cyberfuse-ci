"""
knowledge_graph/gat_link_predictor.py

Graph Attention Network (GAT) link predictor for the CyberFuse-CI knowledge graph.

Combines TransH-based graph embeddings with BERT-derived textual node vectors
to predict missing edges in the vulnerability knowledge graph. This implements
the structural link prediction component described in Layer 2 of the paper.

The GAT receives a node feature matrix where each node's features are the
concatenation of its TransH embedding and its BERT description embedding.
It outputs edge confidence scores for candidate (subject, predicate, object) triples.

Architecture:
    Node features (TransH + BERT) -> GATConv layer 1 -> GATConv layer 2 -> MLP -> sigmoid score

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimensions used in the paper
TRANSHE_DIM  = 128   # TransH entity embedding dimension
BERT_DIM     = 768   # BERT CLS embedding dimension (bert-base-uncased)
GAT_HIDDEN   = 256   # GAT hidden layer size
GAT_HEADS    = 4     # Number of attention heads


class GATLinkPredictor:
    """
    GAT-based link predictor for the vulnerability knowledge graph.

    This class wraps the PyTorch Geometric GAT implementation.
    The model is trained on the initial knowledge graph (positive edges
    from NER/relation extraction) and negative samples (non-edges).

    Args:
        transhe_dim:    Dimension of TransH entity embeddings.
        bert_dim:       Dimension of BERT node text embeddings.
        gat_hidden:     GAT hidden layer dimension.
        gat_heads:      Number of GAT attention heads.
        threshold:      Minimum edge confidence score to assert a new link.
        device:         Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        transhe_dim: int = TRANSHE_DIM,
        bert_dim: int = BERT_DIM,
        gat_hidden: int = GAT_HIDDEN,
        gat_heads: int = GAT_HEADS,
        threshold: float = 0.7,
        device: str = "cpu",
    ):
        self.transhe_dim = transhe_dim
        self.bert_dim    = bert_dim
        self.gat_hidden  = gat_hidden
        self.gat_heads   = gat_heads
        self.threshold   = threshold
        self.device_str  = device
        self._model      = None
        self._trained    = False

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GATConv

            class GATLinkModel(nn.Module):
                def __init__(self, in_dim, hidden, heads, out_dim=1):
                    super().__init__()
                    self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=0.2)
                    self.conv2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
                    self.mlp   = nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden, out_dim),
                        nn.Sigmoid(),
                    )
                    self.act = nn.ELU()

                def forward(self, x, edge_index, candidate_edges):
                    h = self.act(self.conv1(x, edge_index))
                    h = self.act(self.conv2(h, edge_index))
                    src, dst = candidate_edges
                    edge_feat = torch.cat([h[src], h[dst]], dim=-1)
                    return self.mlp(edge_feat).squeeze(-1)

            in_dim = self.transhe_dim + self.bert_dim
            return GATLinkModel(in_dim, self.gat_hidden, self.gat_heads)

        except ImportError:
            raise ImportError(
                "Install torch-geometric: pip install torch-geometric\n"
                "Also ensure torch is installed for your CUDA version."
            )

    def train(
        self,
        node_features: np.ndarray,
        pos_edges: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> list[float]:
        """
        Train the GAT on positive edges (from knowledge graph) and random negative edges.

        Args:
            node_features:  (n_nodes, transhe_dim + bert_dim) numpy array.
            pos_edges:      (2, n_edges) numpy array of (source, target) positive edge pairs.
            epochs:         Training epochs.
            lr:             Learning rate.

        Returns:
            List of training losses per epoch.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("Install torch: pip install torch")

        device = torch.device(self.device_str)
        self._model = self._build_model().to(device)

        X     = torch.tensor(node_features, dtype=torch.float32).to(device)
        edges = torch.tensor(pos_edges, dtype=torch.long).to(device)

        n_nodes = X.shape[0]
        n_pos   = pos_edges.shape[1]

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        losses = []

        self._model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Negative sampling: random (src, dst) pairs not in pos_edges
            neg_src = np.random.randint(0, n_nodes, n_pos)
            neg_dst = np.random.randint(0, n_nodes, n_pos)
            neg_edges = torch.tensor(
                np.stack([neg_src, neg_dst]), dtype=torch.long
            ).to(device)

            all_edges    = torch.cat([edges, neg_edges], dim=1)
            pos_labels   = torch.ones(n_pos, device=device)
            neg_labels   = torch.zeros(n_pos, device=device)
            all_labels   = torch.cat([pos_labels, neg_labels])

            scores = self._model(X, edges, all_edges)
            loss   = criterion(scores, all_labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d  Loss=%.4f", epoch + 1, epochs, loss.item())

        self._trained = True
        return losses

    def predict_links(
        self,
        node_features: np.ndarray,
        existing_edges: np.ndarray,
        candidate_pairs: Optional[np.ndarray] = None,
        max_candidates: int = 5000,
    ) -> list[dict]:
        """
        Predict high-confidence missing links in the knowledge graph.

        Args:
            node_features:    (n_nodes, feature_dim) node embeddings.
            existing_edges:   (2, n_edges) existing positive edges.
            candidate_pairs:  (2, n_candidates) optional candidate edge pairs.
                              If None, samples random pairs excluding existing edges.
            max_candidates:   Maximum number of candidate pairs to evaluate.

        Returns:
            List of dicts: {src_idx, dst_idx, confidence} for edges above threshold.
        """
        if not self._trained or self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("Install torch: pip install torch")

        device = torch.device(self.device_str)
        self._model.eval()

        X     = torch.tensor(node_features, dtype=torch.float32).to(device)
        edges = torch.tensor(existing_edges, dtype=torch.long).to(device)
        n_nodes = X.shape[0]

        if candidate_pairs is None:
            existing_set = set(zip(existing_edges[0].tolist(), existing_edges[1].tolist()))
            src = np.random.randint(0, n_nodes, max_candidates)
            dst = np.random.randint(0, n_nodes, max_candidates)
            mask = np.array([
                (src[i], dst[i]) not in existing_set and src[i] != dst[i]
                for i in range(max_candidates)
            ])
            src, dst = src[mask], dst[mask]
            candidate_pairs = np.stack([src, dst])

        cands = torch.tensor(candidate_pairs, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = self._model(X, edges, cands).cpu().numpy()

        results = []
        for i, (score, (s, d)) in enumerate(zip(scores, zip(candidate_pairs[0], candidate_pairs[1]))):
            if score >= self.threshold:
                results.append({
                    "src_idx":    int(s),
                    "dst_idx":    int(d),
                    "confidence": float(score),
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
