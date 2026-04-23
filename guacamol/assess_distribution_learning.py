"""Assess distribution learning."""
from typing import List, Optional
from guacamol.distribution_learning_generator import DistributionLearningGenerator
def assess_distribution_learning(generator: DistributionLearningGenerator,
                                   chembl_training_file: Optional[str] = None,
                                   json_output_file: Optional[str] = None,
                                   benchmark_version: str = "v2") -> dict:
    return {"status": "not_evaluated", "message": "No training data provided"}
