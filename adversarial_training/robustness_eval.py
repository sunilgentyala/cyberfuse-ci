"""
adversarial_training/robustness_eval.py

Evaluates a trained CyberFuse-CI classifier under FGSM adversarial perturbation.
Reports clean accuracy, adversarial accuracy, and the accuracy gap across epsilon values.

Usage:
    python adversarial_training/robustness_eval.py \
        --model checkpoints/xgboost_classifier.pkl \
        --data data/cvefixes_features.csv \
        --epsilon 0.005 0.01 0.02 0.05 \
        --output evaluation/results/adversarial_report.csv

Authors: Sunil Gentyala et al.
Paper:   CyberFuse-CI, ICETCI 2026
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from adversarial_training.adversarial_trainer import AdversarialTrainer, FEATURE_NAMES, CONTINUOUS_FEATURE_INDICES
from adversarial_training.fgsm_attack import FGSMAttack

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """
    Measures classification accuracy under FGSM perturbation at multiple epsilon values.

    This produces the adversarial accuracy column in Table II of the paper.
    The paper's reported 91.2% adversarial accuracy corresponds to epsilon=0.01.

    Args:
        trainer:    A loaded AdversarialTrainer instance.
        epsilons:   List of epsilon values to test.
    """

    def __init__(self, trainer: AdversarialTrainer, epsilons: list[float] = None):
        self.trainer = trainer
        self.epsilons = epsilons or [0.005, 0.01, 0.02, 0.05]

    def evaluate(self, X: np.ndarray, y_true_labels: list[str]) -> list[dict]:
        """
        Run clean evaluation and FGSM evaluation at each epsilon.

        Args:
            X:              Feature matrix (n_samples, 47)
            y_true_labels:  Ground truth severity tier strings

        Returns:
            List of result dicts, one per epsilon value plus one for clean baseline.
        """
        y_enc = self.trainer.label_encoder.transform(y_true_labels)
        results = []

        # Clean baseline
        y_pred = self.trainer.predict(X)
        clean_acc = accuracy_score(y_true_labels, y_pred)
        clean_f1  = f1_score(y_enc, self.trainer.label_encoder.transform(y_pred), average="weighted")

        results.append({
            "epsilon":  "clean",
            "accuracy": round(clean_acc * 100, 2),
            "f1_score": round(clean_f1 * 100, 2),
            "acc_drop": 0.0,
        })

        logger.info("Clean accuracy: %.2f%%  F1: %.2f%%", clean_acc * 100, clean_f1 * 100)

        for eps in self.epsilons:
            attack = FGSMAttack(
                epsilon=eps,
                continuous_indices=CONTINUOUS_FEATURE_INDICES,
            )
            X_adv = attack.perturb_numpy(
                predict_fn=self.trainer.predict_proba,
                X=X,
                y=y_enc,
            )
            y_pred_adv = self.trainer.predict(X_adv)
            adv_acc = accuracy_score(y_true_labels, y_pred_adv)
            adv_f1  = f1_score(y_enc, self.trainer.label_encoder.transform(y_pred_adv), average="weighted")

            results.append({
                "epsilon":  eps,
                "accuracy": round(adv_acc * 100, 2),
                "f1_score": round(adv_f1 * 100, 2),
                "acc_drop": round((clean_acc - adv_acc) * 100, 2),
            })

            logger.info("epsilon=%.3f  Adv accuracy: %.2f%%  Drop: %.2f%%", eps, adv_acc * 100, (clean_acc - adv_acc) * 100)

        return results

    def save_csv(self, results: list[dict], path: str):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epsilon", "accuracy", "f1_score", "acc_drop"])
            writer.writeheader()
            writer.writerows(results)
        logger.info("Adversarial robustness report written to %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation for CyberFuse-CI")
    parser.add_argument("--model", required=True, help="Path to saved AdversarialTrainer pickle")
    parser.add_argument("--data", required=True, help="Path to feature CSV with 47 features + label column")
    parser.add_argument("--label-col", default="severity_tier", help="Label column name in CSV")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--output", default="evaluation/results/adversarial_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[args.label_col].tolist()

    trainer = AdversarialTrainer.load(args.model)
    evaluator = RobustnessEvaluator(trainer=trainer, epsilons=args.epsilon)
    results = evaluator.evaluate(X, y)
    evaluator.save_csv(results, args.output)


if __name__ == "__main__":
    main()
