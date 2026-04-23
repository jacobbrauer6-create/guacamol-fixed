"""Distribution learning benchmark."""
from typing import List
class DistributionLearningBenchmark:
    def __init__(self, name: str, benchmark_fn):
        self.name = name
        self.benchmark_fn = benchmark_fn

def distribution_learning_benchmark_suite(version_name: str = "v2") -> List[DistributionLearningBenchmark]:
    return []
