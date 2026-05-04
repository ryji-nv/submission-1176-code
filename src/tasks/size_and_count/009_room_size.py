"""Room-size question curator: floor area of the room."""

from __future__ import annotations


def curate_room_size_questions(
    camera_pose: dict,
    room_bounds: list[float],
) -> list[dict]:
    """
    Ask for the floor area of the room.
    room_bounds: [x_min, y_min, x_max, y_max, z_max]
    """
    if not room_bounds or len(room_bounds) < 4:
        return []
    x_min, y_min, x_max, y_max = room_bounds[:4]
    area = round((x_max - x_min) * (y_max - y_min), 2)
    return [
        {
            "type": "room_size",
            "question": "What is the size of this room (in square meters)?",
            "answer": str(area),
            "camera_name": camera_pose["name"],
            "camera_position": list(camera_pose["position"]),
            "camera_look_at": list(camera_pose["look_at"]),
            "room_bounds": list(room_bounds),
        }
    ]
