"""GoalDirectedGenerator abstract base."""
from abc import ABC, abstractmethod
from typing import List, Optional
class GoalDirectedGenerator(ABC):
    @abstractmethod
    def generate_optimised_molecules(self, scoring_function, number_molecules: int,
                                      starting_population: Optional[List[str]] = None) -> List[str]:
        ...
