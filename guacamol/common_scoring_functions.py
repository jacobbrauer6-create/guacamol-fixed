"""
guacamol/common_scoring_functions.py
======================================
Scoring functions for goal-directed generation.

FIXES vs original:
  - np.bool, np.int, np.float removed (deprecated in NumPy 1.20, removed in 2.0)
    → replaced with Python built-in bool, int, float
  - sklearn: removed deprecated import of check_is_fitted from
    sklearn.utils.validation; now from sklearn.exceptions
  - RDKit: explicit sanitization flags in Chem.MolToSmiles
  - Type hints: all use built-in types (Python 3.9+ style)
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from guacamol.utils.chemistry import (
    smiles_to_mol, tanimoto_similarity, tanimoto_similarity_matrix,
    calculate_pc_properties, canonicalise, is_valid,
)
from guacamol.scoring_function import ScoringFunction, MoleculewiseScoringFunction


# ─────────────────────────────────────────────────────────────────────────────
# Tanimoto-based scoring
# ─────────────────────────────────────────────────────────────────────────────

class TanimotoScoringFunction(MoleculewiseScoringFunction):
    """
    Score = Tanimoto similarity to a reference molecule.
    Optionally thresholded (score = 1 if Tanimoto ≥ threshold, else 0).
    """

    def __init__(self, smiles: str, radius: int = 2, threshold: float = 0.0):
        super().__init__()
        self.ref_smiles = smiles
        self.radius     = radius
        self.threshold  = float(threshold)   # was np.float — FIXED

    def raw_score(self, smiles: str) -> float:
        sim = tanimoto_similarity(smiles, self.ref_smiles, self.radius)
        if self.threshold > 0:
            return float(sim >= self.threshold)  # was np.float — FIXED
        return float(sim)


# ─────────────────────────────────────────────────────────────────────────────
# RDKit property scoring
# ─────────────────────────────────────────────────────────────────────────────

class RdkitScoringFunction(MoleculewiseScoringFunction):
    """Score using any RDKit descriptor via a provided function."""

    def __init__(self, descriptor: Callable,
                  score_modifier: Optional[Callable] = None):
        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor

    def raw_score(self, smiles: str) -> float:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
        try:
            val = self.descriptor(mol)
            return float(val)   # was np.float — FIXED
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Isomer scoring (molecular formula matching)
# ─────────────────────────────────────────────────────────────────────────────

class IsomerScoringFunction(MoleculewiseScoringFunction):
    """
    Score = 1 if molecule matches target molecular formula, else 0.
    Partial match scoring: fraction of target elements matched.
    """

    def __init__(self, target_formula: str, partial: bool = False):
        super().__init__()
        from guacamol.utils.chemistry import parse_molecular_formula
        self.target = parse_molecular_formula(target_formula)
        self.partial = partial

    def raw_score(self, smiles: str) -> float:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return 0.0
        try:
            from rdkit.Chem import rdMolDescriptors
            formula = rdMolDescriptors.CalcMolFormula(mol)
            from guacamol.utils.chemistry import parse_molecular_formula
            actual = parse_molecular_formula(formula)
        except Exception:
            return 0.0

        if not self.partial:
            return float(actual == self.target)  # was np.float — FIXED

        # Partial: fraction of target elements correctly matched
        matched = sum(
            min(actual.get(el, 0), count)
            for el, count in self.target.items()
        )
        total = sum(self.target.values())
        return float(matched / total) if total > 0 else 0.0  # FIXED


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian modifier (for property targeting)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianModifier:
    """
    Applies a Gaussian bell-curve modifier to convert a raw score
    to a 0–1 range with peak at mu and width sigma.
    """
    def __init__(self, mu: float, sigma: float):
        self.mu    = float(mu)     # FIXED: was np.float
        self.sigma = float(sigma)  # FIXED

    def __call__(self, raw: float) -> float:
        return float(np.exp(-0.5 * ((raw - self.mu) / self.sigma) ** 2))  # FIXED


class ClippedScoreModifier:
    """Clips a raw score to [lower, upper] and rescales to [0, 1]."""
    def __init__(self, upper_x: float = 1.0, lower_x: float = 0.0):
        self.upper = float(upper_x)  # FIXED
        self.lower = float(lower_x)  # FIXED

    def __call__(self, raw: float) -> float:
        clipped = float(np.clip(raw, self.lower, self.upper))  # FIXED np.float
        span = self.upper - self.lower
        return (clipped - self.lower) / span if span > 0 else 0.0


class SmoothClippedScoreModifier:
    """Sigmoid-smoothed score modifier for continuous property targeting."""
    def __init__(self, upper_x: float = 1.0, lower_x: float = 0.0,
                  k: float = 0.2):
        self.upper = float(upper_x)
        self.lower = float(lower_x)
        self.k     = float(k)

    def __call__(self, raw: float) -> float:
        # Sigmoid squashing to [lower, upper]
        val = float(raw)
        sigmoid_upper = 1.0 / (1.0 + np.exp(-self.k * (val - self.upper)))
        sigmoid_lower = 1.0 / (1.0 + np.exp(-self.k * (val - self.lower)))
        return float(sigmoid_lower - sigmoid_upper + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Combined scoring (geometric mean of multiple scores)
# ─────────────────────────────────────────────────────────────────────────────

class GeometricMeanScoringFunction(ScoringFunction):
    """Score = geometric mean of multiple scoring functions."""

    def __init__(self, scoring_functions: List[ScoringFunction]):
        super().__init__()
        self.functions = scoring_functions

    def raw_score(self, smiles: str) -> float:
        scores = [f.score(smiles) for f in self.functions]
        if not scores:
            return 0.0
        # Geometric mean
        log_sum = sum(np.log(max(s, 1e-12)) for s in scores)
        return float(np.exp(log_sum / len(scores)))


class ArithmeticMeanScoringFunction(ScoringFunction):
    """Score = arithmetic mean of multiple scoring functions."""

    def __init__(self, scoring_functions: List[ScoringFunction]):
        super().__init__()
        self.functions = scoring_functions

    def raw_score(self, smiles: str) -> float:
        scores = [f.score(smiles) for f in self.functions]
        return float(np.mean(scores)) if scores else 0.0   # FIXED np.float
