"""DistributionLearningGenerator abstract base."""
from abc import ABC, abstractmethod
from typing import List
class DistributionLearningGenerator(ABC):
    @abstractmethod
    def generate(self, number_samples: int) -> List[str]: ...
    @abstractmethod
    def train(self, training_set: List[str]) -> None: ...
