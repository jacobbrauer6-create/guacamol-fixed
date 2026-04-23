"""Assess goal-directed generation."""
from typing import List, Optional
from guacamol.goal_directed_generator import GoalDirectedGenerator
def assess_goal_directed_generation(generator: GoalDirectedGenerator,
                                     benchmark_version: str = "v2",
                                     json_output_file: Optional[str] = None) -> dict:
    return {"status": "not_evaluated"}
