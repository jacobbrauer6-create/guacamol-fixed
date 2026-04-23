"""
GuacaMol — community-patched fork (0.5.5.post1)
  NumPy 2.x / Python 3.10–3.12 compatible
  rdkit-pypi pin removed (accepts rdkit >=2022.3)
  np.bool/np.int/np.float aliases replaced with builtins
  scikit-learn 1.x API fixes applied
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("guacamol")
    except PackageNotFoundError:
        __version__ = "0.5.5.post1"
except ImportError:
    __version__ = "0.5.5.post1"

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.benchmark_suites import (
    goal_directed_benchmark_suite,
    distribution_learning_benchmark_suite,
)
from guacamol.common_scoring_functions import (
    TanimotoScoringFunction,
    RdkitScoringFunction,
    IsomerScoringFunction,
)
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.distribution_learning_generator import DistributionLearningGenerator

__all__ = [
    "__version__",
    "assess_distribution_learning",
    "assess_goal_directed_generation",
    "goal_directed_benchmark_suite",
    "distribution_learning_benchmark_suite",
    "TanimotoScoringFunction",
    "RdkitScoringFunction",
    "IsomerScoringFunction",
    "GoalDirectedGenerator",
    "DistributionLearningGenerator",
]
