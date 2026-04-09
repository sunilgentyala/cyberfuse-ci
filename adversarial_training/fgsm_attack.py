"""
adversarial_training/fgsm_attack.py

Fast Gradient Sign Method (FGSM) perturbation for CyberFuse-CI's
adversarial robustness training (Layer 3).

Goodfellow et al. (2015) introduced FGSM as the simplest single-step
adversarial attack: perturb each input feature by epsilon in the direction
that maximizes the loss gradient. Training on a mix of clean and FGSM-
perturbed samples teaches the classifier to ignore gradient-aligned signals
that adversarial actors deliberately craft.

Reference:
    Goodfellow, I. J., Shlens, J., and Szegedy, C. (2015).
    Explaining and Harnessing Adversarial Examples. ICLR 2015.
    arXiv:1412.6572

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FGSMAttack:
    """
    FGSM adversarial perturbation for tabular feature vectors.

    CyberFuse-CI operates on a 47-dimensional feature vector per vulnerability
    entity. Features include CVSS sub-scores, CWE category embeddings, graph
    centrality measures, STIX indicator temporal recency, and cross-source
    co-occurrence counts. FGSM perturbs the continuous features by epsilon
    in the gradient direction while leaving integer and categorical features
    unchanged.

    Args:
        epsilon:            Perturbation magnitude. Paper used epsilon=0.01.
        continuous_indices: Indices of continuous features to perturb.
                            Categorical and integer features at other indices
                            are left unchanged.
        clip_min:           Minimum feature value after perturbation.
        clip_max:           Maximum feature value after perturbation.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        continuous_indices: Optional[list[int]] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.continuous_indices = continuous_indices
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Apply FGSM perturbation to a batch of feature vectors.

        Args:
            model:      PyTorch model. Must be differentiable w.r.t. inputs.
            X:          Input tensor of shape (batch_size, n_features).
            y:          True labels of shape (batch_size,).
            loss_fn:    Loss function, e.g. nn.CrossEntropyLoss().

        Returns:
            X_adv:      Perturbed tensor of same shape as X.
        """
        X_adv = X.clone().detach().requires_grad_(True)
        output = model(X_adv)
        loss = loss_fn(output, y)
        model.zero_grad()
        loss.backward()

        grad_sign = X_adv.grad.data.sign()

        if self.continuous_indices is not None:
            mask = torch.zeros_like(X_adv)
            mask[:, self.continuous_indices] = 1.0
            grad_sign = grad_sign * mask

        X_adv = X_adv + self.epsilon * grad_sign
        X_adv = X_adv.clamp(self.clip_min, self.clip_max).detach()
        return X_adv

    def perturb_numpy(
        self,
        predict_fn,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn=None,
    ) -> np.ndarray:
        """
        Numpy-compatible FGSM for use with sklearn or XGBoost models.

        Since XGBoost is not differentiable via autograd, this method
        approximates the gradient using finite differences. This is a
        black-box approximation, not a true white-box attack. For white-box
        adversarial evaluation, use the PyTorch path above with a surrogate
        neural network.

        Args:
            predict_fn:     Callable(X) -> probabilities array (batch, n_classes).
            X:              Feature array of shape (batch_size, n_features).
            y:              True class labels (batch_size,).
            loss_fn:        Ignored (finite differences used instead).

        Returns:
            X_adv:          Perturbed array of same shape as X.
        """
        X_adv = X.copy().astype(np.float64)
        delta = 1e-4

        for i in range(X.shape[0]):
            grad = np.zeros(X.shape[1])
            probs = predict_fn(X_adv[i:i+1])
            true_class = int(y[i])
            base_loss = -np.log(probs[0, true_class] + 1e-10)

            indices = self.continuous_indices if self.continuous_indices else range(X.shape[1])
            for j in indices:
                X_temp = X_adv[i:i+1].copy()
                X_temp[0, j] += delta
                probs_perturbed = predict_fn(X_temp)
                perturbed_loss = -np.log(probs_perturbed[0, true_class] + 1e-10)
                grad[j] = (perturbed_loss - base_loss) / delta

            X_adv[i] += self.epsilon * np.sign(grad)
            if self.continuous_indices is not None:
                mask = np.zeros(X.shape[1])
                mask[self.continuous_indices] = 1.0
                X_adv[i] = X[i] + (X_adv[i] - X[i]) * mask

            X_adv[i] = np.clip(X_adv[i], self.clip_min, self.clip_max)

        return X_adv
