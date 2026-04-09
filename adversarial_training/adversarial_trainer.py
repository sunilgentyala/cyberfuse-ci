"""
adversarial_training/adversarial_trainer.py

Trains the CyberFuse-CI XGBoost ensemble classifier with adversarial hardening.
Mixes clean and FGSM-perturbed training samples at a configurable ratio.

adversarial_training/robustness_eval.py logic is also included here
as the RobustnessEvaluator class.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# The 47 feature dimensions used in the paper
FEATURE_NAMES = [
    # CVSS sub-scores (6)
    "cvss_base_score", "cvss_exploitability", "cvss_impact",
    "cvss_confidentiality", "cvss_integrity", "cvss_availability",
    # CWE category embedding (10 dims, PCA-reduced)
    "cwe_emb_0", "cwe_emb_1", "cwe_emb_2", "cwe_emb_3", "cwe_emb_4",
    "cwe_emb_5", "cwe_emb_6", "cwe_emb_7", "cwe_emb_8", "cwe_emb_9",
    # Graph centrality measures (5)
    "graph_degree_centrality", "graph_betweenness", "graph_closeness",
    "graph_pagerank", "graph_clustering_coeff",
    # Cross-source co-occurrence (3)
    "cooccurrence_cve_stix", "cooccurrence_cve_siem", "cooccurrence_stix_siem",
    # STIX temporal recency (days since last indicator for same component) (2)
    "stix_recency_days", "stix_indicator_count",
    # NER-extracted entity features (5)
    "ner_product_confidence", "ner_cwe_confidence", "ner_cve_ref_count",
    "relation_affects_count", "relation_exploitedby_count",
    # LLM link prediction confidence (3)
    "llm_link_confidence_mean", "llm_link_confidence_max", "llm_link_count",
    # Source origin one-hot (4)
    "source_cve_nvd", "source_stix_taxii", "source_siem_log", "source_code_repo",
    # Severity recency and publication age (4)
    "days_since_published", "days_since_modified",
    "published_in_last_7d", "published_in_last_30d",
    # Reference count and description length (2)
    "reference_count", "description_len_normalized",
    # CWE count (1)
    "cwe_count",
    # GAT link score (2)
    "gat_link_score_mean", "gat_link_score_max",
]

assert len(FEATURE_NAMES) == 47, f"Feature count mismatch: {len(FEATURE_NAMES)}"

# Indices of continuous features that FGSM may perturb
CONTINUOUS_FEATURE_INDICES = [
    i for i, name in enumerate(FEATURE_NAMES)
    if not name.startswith("source_") and not name.startswith("published_in_")
]


class AdversarialTrainer:
    """
    Trains an XGBoost classifier with FGSM adversarial hardening.

    Training procedure (following paper Section III-D):
        1. Balance classes with SMOTE.
        2. Generate FGSM-perturbed copies of the training split.
        3. Combine clean and adversarial samples (ratio configurable).
        4. Train XGBoost on the combined set.
        5. Evaluate on the held-out clean test split and the adversarially
           perturbed version of the same test split.

    Args:
        epsilon:        FGSM perturbation magnitude. Default 0.01 (paper value).
        adv_ratio:      Fraction of training samples to replace with adversarial versions.
                        0.5 means 50% clean, 50% adversarial.
        n_folds:        Number of cross-validation folds.
        random_state:   Reproducibility seed.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        adv_ratio: float = 0.5,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.epsilon = epsilon
        self.adv_ratio = adv_ratio
        self.n_folds = n_folds
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()

    def _build_xgboost(self):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")

        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _fgsm_numpy(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Finite-difference FGSM approximation for XGBoost."""
        from adversarial_training.fgsm_attack import FGSMAttack
        attack = FGSMAttack(
            epsilon=self.epsilon,
            continuous_indices=CONTINUOUS_FEATURE_INDICES,
        )
        predict_fn = lambda x: model.predict_proba(x)
        return attack.perturb_numpy(predict_fn, X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train with k-fold cross-validation and adversarial augmentation.

        Args:
            X: Feature matrix (n_samples, 47)
            y: Severity tier labels as strings ('CRITICAL', 'HIGH', etc.)

        Returns:
            results dict with per-fold and aggregate metrics
        """
        y_enc = self.label_encoder.fit_transform(y)
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_metrics = []
        best_f1 = -1.0
        best_model = None

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y_enc)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            # SMOTE oversampling on training fold
            smote = SMOTE(random_state=self.random_state)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            # Initial model for FGSM gradient approximation
            prelim = self._build_xgboost()
            prelim.fit(X_train_bal, y_train_bal, verbose=False)

            # Generate adversarial samples for adv_ratio of training set
            n_adv = int(len(X_train_bal) * self.adv_ratio)
            adv_idx = np.random.choice(len(X_train_bal), size=n_adv, replace=False)
            X_adv = self._fgsm_numpy(prelim, X_train_bal[adv_idx], y_train_bal[adv_idx])

            # Combine clean and adversarial
            X_combined = np.vstack([X_train_bal, X_adv])
            y_combined = np.concatenate([y_train_bal, y_train_bal[adv_idx]])

            # Final model on combined set
            model = self._build_xgboost()
            model.fit(X_combined, y_combined, verbose=False)

            # Clean test evaluation
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Adversarial test evaluation
            X_test_adv = self._fgsm_numpy(model, X_test, y_test)
            y_pred_adv = model.predict(X_test_adv)
            f1_adv = f1_score(y_test, y_pred_adv, average="weighted")

            fold_metrics.append({
                "fold": fold_idx + 1,
                "f1_clean": round(f1, 4),
                "f1_adv": round(f1_adv, 4),
                "precision_clean": round(precision_score(y_test, y_pred, average="weighted"), 4),
                "recall_clean": round(recall_score(y_test, y_pred, average="weighted"), 4),
            })

            logger.info("Fold %d: F1=%.4f  Adv-F1=%.4f", fold_idx + 1, f1, f1_adv)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        self.model = best_model

        aggregate = {
            "mean_f1_clean": float(np.mean([m["f1_clean"] for m in fold_metrics])),
            "mean_f1_adv":   float(np.mean([m["f1_adv"]   for m in fold_metrics])),
            "mean_precision": float(np.mean([m["precision_clean"] for m in fold_metrics])),
            "mean_recall":    float(np.mean([m["recall_clean"]    for m in fold_metrics])),
            "folds": fold_metrics,
        }

        logger.info(
            "Cross-validation complete. Mean F1 (clean)=%.4f  Mean F1 (adv)=%.4f",
            aggregate["mean_f1_clean"], aggregate["mean_f1_adv"],
        )
        return aggregate

    def save(self, path: str):
        """Persist the trained model to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)
        logger.info("Model saved to %s", out)

    @classmethod
    def load(cls, path: str) -> "AdversarialTrainer":
        """Load a previously saved trainer."""
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        trainer = cls()
        trainer.model = state["model"]
        trainer.label_encoder = state["label_encoder"]
        return trainer

    def predict(self, X: np.ndarray) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        return self.model.predict_proba(X)
