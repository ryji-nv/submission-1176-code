"""Route-planning question curator: navigate between objects via turns."""

from __future__ import annotations

import math
import random

from src.tasks import object_label, _MAX_TRIALS
from src.tasks.templates import pick_template


def _compute_turn(
    pos: tuple,
    facing: tuple,
    target: tuple,
) -> str | None:
    """Compute the turn needed at pos (facing a direction) to face target.
    Returns 'Turn Left', 'Turn Right', 'Turn Back', or None if already facing."""
    fx = facing[0] - pos[0]
    fy = facing[1] - pos[1]
    flen = math.sqrt(fx * fx + fy * fy)
    if flen < 1e-9:
        return None
    fx, fy = fx / flen, fy / flen
    tx = target[0] - pos[0]
    ty = target[1] - pos[1]
    tlen = math.sqrt(tx * tx + ty * ty)
    if tlen < 1e-9:
        return None
    tx, ty = tx / tlen, ty / tlen
    dot = fx * tx + fy * ty
    cross = fx * ty - fy * tx
    angle = math.degrees(math.atan2(cross, dot))
    # Only accept clear directions: within 30° of 90°/180°/-90°
    if -120 < angle < -60:
        return "Turn Right"
    elif 60 < angle < 120:
        return "Turn Left"
    elif abs(angle) > 150:
        return "Turn Back"
    return None


def curate_route_planning_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """
    Navigate from object A (facing B) to object C via turns.
    Matches VSI-Bench route_planning format.
    """
    if len(visible_objects) < 3:
        return []

    indices = list(range(len(visible_objects)))
    result: list[dict] = []
    used: set[tuple] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            i, j, k = rng.sample(indices, 3)
            key = (i, j, k)
            if key in used:
                continue
            start = visible_objects[i]
            facing = visible_objects[j]
            dest = visible_objects[k]
            labels = {object_label(start), object_label(facing), object_label(dest)}
            if len(labels) < 3:
                continue
            sp = (start["centroid"][0], start["centroid"][1])
            fp = (facing["centroid"][0], facing["centroid"][1])
            dp = (dest["centroid"][0], dest["centroid"][1])
            # Require objects to be well-separated
            d_sf = math.sqrt((sp[0] - fp[0]) ** 2 + (sp[1] - fp[1]) ** 2)
            d_sd = math.sqrt((sp[0] - dp[0]) ** 2 + (sp[1] - dp[1]) ** 2)
            d_fd = math.sqrt((fp[0] - dp[0]) ** 2 + (fp[1] - dp[1]) ** 2)
            if min(d_sf, d_sd, d_fd) < 1.0:
                continue
            turn = _compute_turn(sp, fp, dp)
            if turn is None:
                continue
            used.add(key)
            found = (start, facing, dest, turn)
            break

        if found:
            start, facing, dest, turn = found
            sl = object_label(start)
            fl = object_label(facing)
            dl = object_label(dest)
            _TURN_OPTIONS = ["Turn Left", "Turn Right", "Turn Back"]
            wrong = [t for t in _TURN_OPTIONS if t != turn]
            raw = [turn] + wrong
            rng.shuffle(raw)
            choices = [f"{chr(65 + i)}. {c}" for i, c in enumerate(raw)]
            answer = chr(65 + raw.index(turn))

            question = pick_template(
                "route_planning", rng, sl=sl, fl=fl, dl=dl
            )

            result.append(
                {
                    "type": "route_planning",
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "camera_name": camera_pose["name"],
                    "camera_position": list(camera_pose["position"]),
                    "camera_look_at": list(camera_pose["look_at"]),
                    "start_object": object_label(start),
                    "facing_object": object_label(facing),
                    "destination_object": object_label(dest),
                }
            )

    return result
