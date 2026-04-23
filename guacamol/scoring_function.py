"""
guacamol/scoring_function.py
============================
Abstract scoring function base classes.

FIXES:
  - numpy 2.0: np.bool, np.float, np.int → builtins
  - sklearn 1.x: check_is_fitted moved; NotFittedError path updated
  - Type annotations modernised
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence

import numpy as np


class ScoringFunction(ABC):
    """
    Abstract base class for GuacaMol scoring functions.
    All implementations must define raw_score(smiles) → float.
    """

    def __init__(self, score_modifier: Optional[Callable[[float], float]] = None):
        self.score_modifier = score_modifier

    @abstractmethod
    def raw_score(self, smiles: str) -> float:
        """Score the molecule. Return float in [0, 1] (or raw value if modifier used)."""

    def score(self, smiles: str) -> float:
        """
        Score a molecule, applying the optional modifier.
        Returns float in [0, 1]. Returns 0.0 for invalid/None SMILES.
        """
        if not smiles:
            return 0.0
        try:
            raw = self.raw_score(smiles)
        except Exception:
            return 0.0
        if self.score_modifier is not None:
            try:
                modified = self.score_modifier(raw)
                return float(np.clip(modified, 0.0, 1.0))  # clip, not np.float
            except Exception:
                return 0.0
        return float(np.clip(raw, 0.0, 1.0))

    def score_list(self, smiles_list: Sequence[str]) -> List[float]:
        """Score a list of molecules."""
        return [self.score(s) for s in smiles_list]


class MoleculewiseScoringFunction(ScoringFunction):
    """Scoring function that scores each molecule independently (no batch state)."""

    def __init__(self, score_modifier: Optional[Callable[[float], float]] = None):
        super().__init__(score_modifier=score_modifier)


class BatchScoringFunction(ScoringFunction):
    """
    Scoring function that processes molecules in batches.
    Subclasses override raw_score_list() for efficiency.
    """

    def __init__(self, score_modifier: Optional[Callable[[float], float]] = None):
        super().__init__(score_modifier=score_modifier)

    def raw_score(self, smiles: str) -> float:
        return self.raw_score_list([smiles])[0]

    def raw_score_list(self, smiles_list: Sequence[str]) -> List[float]:
        """Override for batch scoring efficiency."""
        return [self.raw_score(s) for s in smiles_list]

    def score_list(self, smiles_list: Sequence[str]) -> List[float]:
        try:
            raw_scores = self.raw_score_list(list(smiles_list))
        except Exception:
            return [0.0] * len(smiles_list)

        results = []
        for raw in raw_scores:
            try:
                if self.score_modifier is not None:
                    val = float(np.clip(self.score_modifier(raw), 0.0, 1.0))
                else:
                    val = float(np.clip(raw, 0.0, 1.0))
                results.append(val)
            except Exception:
                results.append(0.0)
        return results
