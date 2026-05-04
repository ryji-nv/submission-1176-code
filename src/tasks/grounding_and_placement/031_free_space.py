"""Free-space question curator: select empty, feasible regions on visible surfaces."""

from __future__ import annotations

import random
from importlib import import_module

curate_object_placement_questions = getattr(
    import_module("src.tasks.grounding_and_placement.017_object_placement"),
    "curate_object_placement_questions",
)


def curate_free_space_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    all_objects: list[dict] | None = None,
    *,
    aux_data_dir: str | None = None,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """Reuse the placement-feasibility solver and expose it as free-space QA."""
    questions = curate_object_placement_questions(
        camera_pose,
        visible_objects,
        all_objects,
        aux_data_dir=aux_data_dir,
        max_questions=max_questions,
        rng=rng,
    )
    for question in questions:
        question["type"] = "free_space"
        question.pop("moveable_object", None)
        question.pop("footprint", None)
    return questions
