"""Shared constants for question-type classification and generation."""

ALL_QUESTION_TYPES = [
    "closest_object",
    "depth_estimation",
    "distance_estimation",
    "object_distance",
    "relative_direction",
    "camera_object_position",
    "depth_ordering",
    "object_size",
    "object_count",
    "room_size",
    "object_grounding",
    "camera_relative_position",
    "camera_facing_direction",
    "camera_motion",
    "viewpoint_change",
    "spatial_imagination",
    "compound_spatial_referring",
    "object_placement",
    "free_space",
    "spatial_compatibility",
    "object_matching_mv",
    "depth_difference",
    "object_grounding_bbox",
    "object_category",
    "object_size_qualitative",
    "comparative_spatial_grounding",
    "ordinal_grounding",
    "object_object_position_mv",
    "camera_region_position",
    "object_region_position",
    "region_region_position",
    "route_planning",
]

DOT_ANNOTATED_TYPES = frozenset(
    {
        "closest_object",
        "depth_estimation",
        "distance_estimation",
        "object_distance",
        "depth_ordering",
        "depth_difference",
    }
)

DUAL_IMAGE_TYPES = frozenset(
    {
        "camera_relative_position",
        "camera_facing_direction",
        "camera_motion",
        "viewpoint_change",
        "object_matching_mv",
        "object_object_position_mv",
    }
)

MULTI_VIEW_TYPES = frozenset(
    {
        "object_object_position_mv",
        "object_matching_mv",
        "viewpoint_change",
        "camera_relative_position",
        "camera_facing_direction",
    }
)

EDGE_CAMERA_TYPES = frozenset(
    {
        "camera_relative_position",
    }
)

NAME_ONLY_TYPES = frozenset(
    {
        "object_size",
        "object_placement",
        "free_space",
        "spatial_compatibility",
        "route_planning",
        "object_region_position",
        "object_grounding_bbox",
        "object_size_qualitative",
    }
)

TASK_DIR_NAME = {t: f"{i:03d}_{t}" for i, t in enumerate(ALL_QUESTION_TYPES)}

MIN_DISTANCE_GAP = 0.5

MAX_TRIALS = 200
