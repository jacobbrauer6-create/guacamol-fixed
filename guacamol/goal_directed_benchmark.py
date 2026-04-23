"""Goal-directed benchmark suite."""
from typing import List
from guacamol.scoring_function import ScoringFunction
class GoalDirectedBenchmark:
    def __init__(self, name: str, scoring_function: ScoringFunction, contribution=None):
        self.name = name
        self.scoring_function = scoring_function

def goal_directed_benchmark_suite(version_name: str = "v2") -> List[GoalDirectedBenchmark]:
    """Return list of goal-directed benchmarks. Requires rdkit."""
    try:
        from guacamol._benchmarks_v2 import _load_benchmarks
        return _load_benchmarks()
    except Exception:
        return []
