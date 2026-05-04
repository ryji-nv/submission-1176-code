"""Camera-motion question curator: identify movement direction from sequential images."""

from __future__ import annotations

import random

from src.tasks import _relative_position_in_frame
from src.tasks.templates import pick_template


_STEP_DIRECTION_ANSWER = {
    "forward": "Forward",
    "left": "Left",
    "right": "Right",
    "back": "Back",
}


def curate_camera_motion_questions(
    base_camera: dict,
    all_cameras: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """
    Pair base_camera with its forward/left/right stepped variants.
    """
    _MOTION_DIRS = {"forward", "left", "right", "back"}
    base_name = base_camera["name"]
    stepped = [c for c in all_cameras
               if c.get("step_from") == base_name
               and c.get("step_direction", "") in _MOTION_DIRS]
    if not stepped:
        return []

    rng.shuffle(stepped)
    result = []
    used: set[str] = set()
    for sc in stepped:
        if len(result) >= max_questions:
            break
        if sc["name"] in used:
            continue
        d = sc.get("step_direction", "")
        pos = _relative_position_in_frame(base_camera, sc)
        if pos is not None:
            t_fwd, t_right, t_up = pos
            _max_val = max(abs(t_fwd), abs(t_right), abs(t_up))
            if _max_val < 0.05:
                continue
            if abs(t_up) == _max_val:
                d = "up" if t_up > 0 else "down"
            elif abs(t_fwd) >= abs(t_right):
                d = "forward" if t_fwd > 0 else "back"
            else:
                d = "right" if t_right > 0 else "left"
        answer = _STEP_DIRECTION_ANSWER.get(d)
        if answer is None:
            continue
        used.add(sc["name"])
        result.append(
            {
                "type": "camera_motion",
                "question": pick_template("camera_motion", rng),
                "choices": ["Forward", "Left", "Right", "Back"],
                "answer": answer,
                "camera_a": {
                    "name": base_camera["name"],
                    "position": list(base_camera["position"]),
                    "look_at": list(base_camera["look_at"]),
                },
                "camera_b": {
                    "name": sc["name"],
                    "position": list(sc["position"]),
                    "look_at": list(sc["look_at"]),
                    "step_direction": d,
                },
            }
        )
    return result
