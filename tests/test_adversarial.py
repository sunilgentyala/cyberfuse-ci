"""
tests/test_adversarial.py

Unit tests for the FGSM adversarial training module.

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

import numpy as np
import pytest

from adversarial_training.fgsm_attack import FGSMAttack
from adversarial_training.adversarial_trainer import FEATURE_NAMES, CONTINUOUS_FEATURE_INDICES


class TestFGSMAttack:
    def _dummy_predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Simple softmax-like mock predict_proba function."""
        n = X.shape[0]
        logits = np.random.rand(n, 5)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def test_perturb_changes_continuous_features(self):
        attack = FGSMAttack(epsilon=0.01, continuous_indices=CONTINUOUS_FEATURE_INDICES)
        X = np.random.rand(10, 47).astype(np.float64)
        y = np.random.randint(0, 5, 10)

        X_adv = attack.perturb_numpy(self._dummy_predict_fn, X, y)
        assert X_adv.shape == X.shape

        # Continuous features should have changed
        continuous_diff = np.abs(X_adv[:, CONTINUOUS_FEATURE_INDICES] - X[:, CONTINUOUS_FEATURE_INDICES])
        assert continuous_diff.sum() > 0

    def test_perturb_respects_clip_bounds(self):
        attack = FGSMAttack(epsilon=0.5, clip_min=0.0, clip_max=1.0)
        X = np.random.rand(5, 47).astype(np.float64)
        y = np.zeros(5, dtype=int)

        X_adv = attack.perturb_numpy(self._dummy_predict_fn, X, y)
        assert X_adv.min() >= 0.0
        assert X_adv.max() <= 1.0

    def test_perturb_categorical_unchanged(self):
        # Source one-hot features (indices 28-31) should not be perturbed when
        # continuous_indices excludes them
        attack = FGSMAttack(epsilon=0.1, continuous_indices=CONTINUOUS_FEATURE_INDICES)
        X = np.random.rand(5, 47).astype(np.float64)
        y = np.zeros(5, dtype=int)

        # Set source flags to known values
        source_indices = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("source_")]
        X[:, source_indices] = [[1, 0, 0, 0]] * 5

        X_adv = attack.perturb_numpy(self._dummy_predict_fn, X, y)

        # Source flags should remain unchanged
        for idx in source_indices:
            np.testing.assert_array_almost_equal(X_adv[:, idx], X[:, idx])

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            FGSMAttack(epsilon=0.0)
        with pytest.raises(ValueError):
            FGSMAttack(epsilon=-0.01)

    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 47, "Paper specifies 47 features"

    def test_continuous_indices_are_valid(self):
        for idx in CONTINUOUS_FEATURE_INDICES:
            assert 0 <= idx < 47, f"Invalid feature index: {idx}"
        # Categorical features (source_ prefix, published_in_ prefix) must NOT be in continuous
        categorical = [i for i, n in enumerate(FEATURE_NAMES)
                       if n.startswith("source_") or n.startswith("published_in_")]
        for cat_idx in categorical:
            assert cat_idx not in CONTINUOUS_FEATURE_INDICES, \
                f"Categorical feature {FEATURE_NAMES[cat_idx]} should not be in continuous_indices"
