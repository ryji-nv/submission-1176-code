"""Question prompt templates — 10 paraphrases per question type.

Each key maps to a list of format-string templates that accept the same
variables.  Use ``pick_template(qtype, rng, **kwargs)`` to randomly select
one and fill in the placeholders.
"""

from __future__ import annotations

import random

_COORD_SUFFIX = (
    " Your answer should be formatted as a list of tuples, i.e. [(x1, y1)],"
    " where each tuple contains the x and y coordinates of a point satisfying"
    " the conditions above. The coordinates should be between 0 and 1,"
    " indicating the normalized pixel locations of the points in the image."
)

_3D_HINT = " Calculate or judge based on the 3D center points of these objects."

TEMPLATES: dict[str, list[str]] = {
    # ------------------------------------------------------------------
    # 000 closest_object
    # vars: label_a, label_b
    # ------------------------------------------------------------------
    "closest_object": [
        "Which is closer to the camera, {label_a} or {label_b}?" + _3D_HINT + " Select the appropriate response from the given choices.",
        "Between {label_a} and {label_b}, which one is nearer to the camera?" + _3D_HINT + " Choose the correct answer.",
        "Comparing {label_a} and {label_b}, which is at a shorter distance from the camera?" + _3D_HINT + " Pick the right option.",
        "Of {label_a} and {label_b}, which is positioned closer to the camera?" + _3D_HINT + " Select your answer from the choices.",
        "Determine which object is closer to the camera: {label_a} or {label_b}." + _3D_HINT + " Choose from the given options.",
        "Which of the following is nearer to the camera — {label_a} or {label_b}?" + _3D_HINT + " Select the correct choice.",
        "Look at {label_a} and {label_b}. Which one appears at a shorter depth from the camera?" + _3D_HINT + " Pick the correct answer.",
        "Identify which object has a smaller distance to the camera: {label_a} or {label_b}." + _3D_HINT + " Choose the right response.",
        "Consider {label_a} and {label_b}. Which is located closer to the viewpoint?" + _3D_HINT + " Select one.",
        "Out of {label_a} and {label_b}, which object is the nearest to the camera?" + _3D_HINT + " Choose the appropriate answer.",
    ],

    # ------------------------------------------------------------------
    # 001 depth_estimation
    # vars: label
    # ------------------------------------------------------------------
    "depth_estimation": [
        "What is the depth of the {label} (red point)?" + _3D_HINT + " The unit is meter. Answer using a single number and nothing else.",
        "How deep is the {label} (red point) from the camera?" + _3D_HINT + " Give the answer in meters as a single number.",
        "Estimate the depth in meters of the {label} (red point)." + _3D_HINT + " Respond with only a number.",
        "What is the distance along the camera's viewing direction to the {label} (red point)?" + _3D_HINT + " Answer in meters, number only.",
        "Determine the depth of the {label} (red point) in meters." + _3D_HINT + " Provide a single numerical answer.",
        "How far along the depth axis is the {label} (red point) from the camera?" + _3D_HINT + " Answer in meters only.",
        "Measure the depth to the {label} (red point) in meters." + _3D_HINT + " Reply with a single number.",
        "Calculate the depth of the {label} (red point) from the camera viewpoint." + _3D_HINT + " State the answer in meters as one number.",
        "What depth value in meters corresponds to the {label} (red point)?" + _3D_HINT + " Answer with a number only.",
        "Report the depth of the {label} (red point) in meters." + _3D_HINT + " Provide only the numerical value.",
    ],

    # ------------------------------------------------------------------
    # 002 distance_estimation
    # vars: label
    # ------------------------------------------------------------------
    "distance_estimation": [
        "What is the total distance in meters from the person who captured the image to the center of {label} (red point)?" + _3D_HINT + " Answer using a single number and nothing else.",
        "How far is the {label} (red point) from the camera in meters?" + _3D_HINT + " Answer with a single number.",
        "Estimate the 3D distance in meters between the camera and the {label} (red point)." + _3D_HINT + " Respond with only a number.",
        "What is the straight-line distance from the camera to the {label} (red point) in meters?" + _3D_HINT + " Give a numerical answer.",
        "Calculate the Euclidean distance in meters from the viewpoint to the {label} (red point)." + _3D_HINT + " Answer with a number only.",
        "Determine how many meters separate the camera from the center of the {label} (red point)." + _3D_HINT + " Provide a single number.",
        "What is the spatial distance from the observer to the {label} (red point) in meters?" + _3D_HINT + " Reply with a number.",
        "Measure the distance from the camera position to the {label} (red point)." + _3D_HINT + " Answer in meters, number only.",
        "How many meters is the {label} (red point) from the image capture point?" + _3D_HINT + " Provide only the numerical value.",
        "Report the distance in meters from the camera to the {label} (red point)." + _3D_HINT + " State a single number.",
    ],

    # ------------------------------------------------------------------
    # 003 object_distance
    # vars: label_a, label_b
    # ------------------------------------------------------------------
    "object_distance": [
        "What is the Euclidean distance between the center of {label_a} (blue point) and {label_b} (red point) in meters?" + _3D_HINT + " Answer using a single number and nothing else.",
        "How far apart are {label_a} (blue point) and {label_b} (red point) in meters?" + _3D_HINT + " Answer with a single number.",
        "Calculate the 3D distance in meters between {label_a} (blue point) and {label_b} (red point)." + _3D_HINT + " Respond with only a number.",
        "What is the straight-line distance between {label_a} (blue point) and {label_b} (red point)?" + _3D_HINT + " Give the answer in meters as one number.",
        "Determine the spatial distance in meters separating {label_a} (blue point) from {label_b} (red point)." + _3D_HINT + " Answer numerically.",
        "Measure the distance from {label_a} (blue point) to {label_b} (red point) in meters." + _3D_HINT + " Provide a single number.",
        "How many meters separate {label_a} (blue point) and {label_b} (red point)?" + _3D_HINT + " Reply with a number only.",
        "Estimate the 3D Euclidean distance between {label_a} (blue point) and {label_b} (red point) in meters." + _3D_HINT + " Answer with one number.",
        "What is the distance in meters between the centers of {label_a} (blue point) and {label_b} (red point)?" + _3D_HINT + " State a single number.",
        "Report the Euclidean distance in meters from {label_a} (blue point) to {label_b} (red point)." + _3D_HINT + " Provide only the numerical value.",
    ],

    # ------------------------------------------------------------------
    # 004 relative_direction
    # vars: label_a, label_b
    # ------------------------------------------------------------------
    "relative_direction": [
        "In the provided image, describe the location of the {label_a} (blue bbox) with respect to the {label_b} (red bbox), considering the observer's viewpoint." + _3D_HINT + " Select the correct response from the given choices.",
        "Where is {label_a} (blue bbox) positioned relative to {label_b} (red bbox) from the camera's perspective?" + _3D_HINT + " Choose the correct answer.",
        "From the observer's point of view, how is {label_a} (blue bbox) situated in relation to {label_b} (red bbox)?" + _3D_HINT + " Pick the right option.",
        "Describe the spatial relationship of {label_a} (blue bbox) relative to {label_b} (red bbox) as seen by the camera." + _3D_HINT + " Select your answer.",
        "Considering the camera's viewpoint, what is the direction of {label_a} (blue bbox) from {label_b} (red bbox)?" + _3D_HINT + " Choose from the options.",
        "Relative to {label_b} (red bbox), where does {label_a} (blue bbox) appear from the observer's view?" + _3D_HINT + " Select the appropriate choice.",
        "How would you describe the position of {label_a} (blue bbox) with respect to {label_b} (red bbox) in this image?" + _3D_HINT + " Pick the correct response.",
        "From the viewer's perspective, in which direction is {label_a} (blue bbox) located relative to {label_b} (red bbox)?" + _3D_HINT + " Choose the right answer.",
        "Looking at the image, what is the spatial position of {label_a} (blue bbox) compared to {label_b} (red bbox)?" + _3D_HINT + " Select one option.",
        "Determine the relative direction of {label_a} (blue bbox) from {label_b} (red bbox) as seen from the camera." + _3D_HINT + " Choose the correct response.",
    ],

    # ------------------------------------------------------------------
    # 005 camera_object_position
    # vars: label
    # ------------------------------------------------------------------
    "camera_object_position": [
        "How is object {label} (bbox) situated with respect to the observer's main viewpoint?" + _3D_HINT + " Select the correct response from the given choices.",
        "Where is {label} (bbox) located relative to the camera's center of view?" + _3D_HINT + " Choose the correct answer.",
        "From the observer's perspective, in which part of the view is {label} (bbox) positioned?" + _3D_HINT + " Pick the right option.",
        "Describe the position of {label} (bbox) relative to the camera's viewing direction." + _3D_HINT + " Select your answer from the choices.",
        "In which region of the observer's field of view does {label} (bbox) appear?" + _3D_HINT + " Choose from the options.",
        "Relative to the camera's forward direction, where is {label} (bbox)?" + _3D_HINT + " Select the appropriate choice.",
        "How is {label} (bbox) positioned in relation to the camera's viewpoint?" + _3D_HINT + " Pick the correct response.",
        "Considering the observer's view, describe the location of {label} (bbox)." + _3D_HINT + " Choose the right answer.",
        "What is the spatial position of {label} (bbox) from the observer's main viewing angle?" + _3D_HINT + " Select one option.",
        "Determine where {label} (bbox) is situated with respect to the camera's line of sight." + _3D_HINT + " Choose the correct response.",
    ],

    # ------------------------------------------------------------------
    # 006 depth_ordering (no vars)
    # ------------------------------------------------------------------
    "depth_ordering": [
        "Order the marked objects from nearest to farthest: blue point, red point, green point.",
        "Rank the following objects by their distance from the camera, closest first: blue point, red point, green point.",
        "Sort the marked objects from closest to most distant: blue point, red point, green point.",
        "Arrange blue point, red point, and green point in order from nearest to farthest from the camera.",
        "List the three marked objects in order of increasing distance from the viewpoint: blue, red, green.",
        "Which ordering correctly places blue point, red point, and green point from nearest to farthest?",
        "Determine the depth order of the three marked objects (blue, red, green) from closest to farthest.",
        "Place blue point, red point, and green point in sequence from the one closest to the camera to the farthest.",
        "From nearest to most distant, what is the correct ordering of blue point, red point, and green point?",
        "Sequence the three colored points (blue, red, green) by their depth, starting with the closest.",
    ],

    # ------------------------------------------------------------------
    # 007 object_size
    # vars: label
    # ------------------------------------------------------------------
    "object_size": [
        "What is the length of the longest dimension (length, width, or height) of the {label}, measured in centimeters?",
        "How long is the largest dimension of the {label} in centimeters?",
        "Determine the longest edge of the {label} in centimeters (length, width, or height).",
        "What is the maximum dimension of the {label} in centimeters?",
        "In centimeters, what is the size of the longest side of the {label}?",
        "Measure the longest dimension (length, width, or height) of the {label}. Answer in centimeters.",
        "Report the largest dimension of the {label} in centimeters.",
        "Among the length, width, and height of the {label}, which is the longest? Answer in centimeters.",
        "What is the greatest extent of the {label} along any axis, in centimeters?",
        "How many centimeters is the longest dimension of the {label}?",
    ],

    # ------------------------------------------------------------------
    # 008 object_count
    # vars: clean_type
    # ------------------------------------------------------------------
    "object_count": [
        "How many {clean_type}(s) are in this room?",
        "Count the number of {clean_type}(s) visible in this room.",
        "How many instances of {clean_type} can you find in the room?",
        "What is the total count of {clean_type}(s) in this scene?",
        "Determine how many {clean_type}(s) are present in the room.",
        "How many {clean_type}(s) can be seen in this image?",
        "State the number of {clean_type}(s) in the room.",
        "In this room, how many {clean_type}(s) are there?",
        "Identify the total number of {clean_type}(s) in this scene.",
        "Report how many {clean_type}(s) exist in the room.",
    ],

    # ------------------------------------------------------------------
    # 009 room_size (no vars)
    # ------------------------------------------------------------------
    "room_size": [
        "What is the size of this room (in square meters)?",
        "How large is this room in square meters?",
        "Estimate the floor area of this room in square meters.",
        "What is the area of this room in square meters?",
        "Determine the room size in square meters.",
        "How many square meters is this room?",
        "Calculate the floor area of this room in square meters.",
        "What is the total area of this room (in m²)?",
        "Report the size of this room in square meters.",
        "In square meters, how big is this room?",
    ],

    # ------------------------------------------------------------------
    # 010 object_grounding
    # vars: label
    # ------------------------------------------------------------------
    "object_grounding": [
        "Please point out the {label}." + _COORD_SUFFIX,
        "Locate the {label} in the image." + _COORD_SUFFIX,
        "Identify where the {label} is in the image." + _COORD_SUFFIX,
        "Show the position of the {label}." + _COORD_SUFFIX,
        "Point to the {label} in this image." + _COORD_SUFFIX,
        "Where is the {label} located in this image?" + _COORD_SUFFIX,
        "Indicate the {label} in the image." + _COORD_SUFFIX,
        "Mark the location of the {label}." + _COORD_SUFFIX,
        "Find and point to the {label}." + _COORD_SUFFIX,
        "Pinpoint the {label} in the image." + _COORD_SUFFIX,
    ],

    # ------------------------------------------------------------------
    # 011 camera_relative_position (no vars)
    # ------------------------------------------------------------------
    "camera_relative_position": [
        "Assuming I am taking the first photo, where is the camera positioned relative to me when taking the second photo?",
        "If the first image is my viewpoint, in which direction is the second camera located relative to me?",
        "From the perspective of the first photo, where is the second camera positioned?",
        "Standing at the first camera's location, in which direction would I find the second camera?",
        "Relative to the first camera's position and facing direction, where is the second camera?",
        "If I am the photographer of the first image, in what direction is the second photo taken from?",
        "Considering the first image as my vantage point, where is the second camera relative to me?",
        "From the first camera's viewpoint, describe the position of the second camera.",
        "Where has the camera moved to in the second photo, relative to the first photo's position?",
        "In which direction is the second camera placed compared to the first camera's location and orientation?",
    ],

    # ------------------------------------------------------------------
    # 012 camera_facing_direction
    # vars: ref_dir
    # ------------------------------------------------------------------
    "camera_facing_direction": [
        "Camera A faces {ref_dir}. Which direction is camera B facing?",
        "If camera A is oriented toward {ref_dir}, what direction does camera B face?",
        "Camera A looks toward {ref_dir}. In which direction is camera B pointing?",
        "Given that camera A faces {ref_dir}, determine camera B's facing direction.",
        "Camera A is directed toward {ref_dir}. What is the facing direction of camera B?",
        "Knowing camera A points {ref_dir}, which way is camera B facing?",
        "Camera A's viewing direction is {ref_dir}. What direction does camera B look toward?",
        "If camera A is facing {ref_dir}, in which cardinal direction is camera B oriented?",
        "Camera A is aimed {ref_dir}. Determine the direction camera B is facing.",
        "With camera A pointing {ref_dir}, identify camera B's facing direction.",
    ],

    # ------------------------------------------------------------------
    # 013 camera_motion (no vars)
    # ------------------------------------------------------------------
    "camera_motion": [
        "The images are taken continuously from a first-person perspective. In which direction are you moving?",
        "These sequential images show a first-person view. What direction is the camera moving?",
        "From these consecutive first-person images, determine the direction of movement.",
        "Looking at these sequential first-person photos, which way is the viewer moving?",
        "Based on these continuous first-person images, identify the direction of motion.",
        "These images capture consecutive moments from a first-person view. Which direction are you heading?",
        "Comparing these sequential first-person images, in which direction has the camera moved?",
        "From the first-person perspective shown in these images, what is the movement direction?",
        "Determine the direction of camera movement based on these consecutive first-person images.",
        "What direction of travel is shown in these sequential first-person images?",
    ],

    # ------------------------------------------------------------------
    # 014 viewpoint_change (no vars — format instructions are fixed)
    # ------------------------------------------------------------------
    "viewpoint_change": [
        "Describe the changes in orientation needed to transition from the first image to the second.",
        "What camera movement and rotation transform the first viewpoint into the second?",
        "Specify the translation and rotation required to go from the first image's viewpoint to the second.",
        "How should the camera move and rotate to change from the first view to the second?",
        "Determine the viewpoint change needed to transition from the first image to the second.",
        "Describe the camera displacement and rotation from the first image to the second.",
        "What movement and rotation take the camera from the first viewpoint to the second?",
        "Calculate the translation and rotation needed to shift from the first view to the second.",
        "Characterize the camera transformation between the first and second images.",
        "Report the camera movement and rotation that maps the first viewpoint to the second.",
    ],

    # ------------------------------------------------------------------
    # 015 spatial_imagination — sub_type "oo"
    # vars: red_label, yellow_label, green_label, blue_label
    # ------------------------------------------------------------------
    "spatial_imagination_oo": [
        "From the observer's point of view, what changes occur in the spatial positioning of {red_label} (red bbox) relative to {yellow_label} (yellow bbox) when observer moves to {green_label} (green bbox) and faces {blue_label} (blue bbox)?" + _3D_HINT + " Select the correct response from the given choices.",
        "If the observer moves to {green_label} (green bbox) and looks toward {blue_label} (blue bbox), how does the position of {red_label} (red bbox) relative to {yellow_label} (yellow bbox) change?" + _3D_HINT + " Choose the correct answer.",
        "Imagine standing at {green_label} (green bbox) and facing {blue_label} (blue bbox). How is {red_label} (red bbox) now positioned relative to {yellow_label} (yellow bbox)?" + _3D_HINT + " Select the right option.",
        "When the observer relocates to {green_label} (green bbox) facing {blue_label} (blue bbox), what is the new spatial relationship of {red_label} (red bbox) to {yellow_label} (yellow bbox)?" + _3D_HINT + " Pick the correct answer.",
        "After moving to {green_label} (green bbox) and orienting toward {blue_label} (blue bbox), describe how {red_label} (red bbox) is positioned compared to {yellow_label} (yellow bbox)." + _3D_HINT + " Choose from the options.",
        "Consider moving to {green_label} (green bbox) and facing {blue_label} (blue bbox). What becomes the relative position of {red_label} (red bbox) with respect to {yellow_label} (yellow bbox)?" + _3D_HINT + " Select your answer.",
        "If you stand at {green_label} (green bbox) looking at {blue_label} (blue bbox), where is {red_label} (red bbox) relative to {yellow_label} (yellow bbox)?" + _3D_HINT + " Choose the correct response.",
        "From the viewpoint at {green_label} (green bbox) facing {blue_label} (blue bbox), how does {red_label} (red bbox) relate spatially to {yellow_label} (yellow bbox)?" + _3D_HINT + " Pick the right response.",
        "Relocate mentally to {green_label} (green bbox) and face {blue_label} (blue bbox). What is the direction of {red_label} (red bbox) from {yellow_label} (yellow bbox)?" + _3D_HINT + " Select one answer.",
        "Standing at {green_label} (green bbox) oriented toward {blue_label} (blue bbox), describe {red_label} (red bbox)'s position relative to {yellow_label} (yellow bbox)." + _3D_HINT + " Choose the appropriate answer.",
    ],

    # ------------------------------------------------------------------
    # 015 spatial_imagination — sub_type "oc"
    # vars: red_label, green_label, blue_label
    # ------------------------------------------------------------------
    "spatial_imagination_oc": [
        "How does the positional relationship of {red_label} (red bbox) to the observer change when the observer moves to the 3D center of {green_label} (green bbox) and faces {blue_label} (blue bbox)?" + _3D_HINT + " Select the correct response from the given choices.",
        "If you move to {green_label} (green bbox) and look toward {blue_label} (blue bbox), where does {red_label} (red bbox) appear relative to you?" + _3D_HINT + " Choose the correct answer.",
        "After relocating to {green_label} (green bbox) facing {blue_label} (blue bbox), in which direction is {red_label} (red bbox) from your new position?" + _3D_HINT + " Select the right option.",
        "Imagine standing at {green_label} (green bbox) oriented toward {blue_label} (blue bbox). Where is {red_label} (red bbox) relative to you?" + _3D_HINT + " Pick the correct answer.",
        "When you move to {green_label} (green bbox) and face {blue_label} (blue bbox), how is {red_label} (red bbox) positioned relative to the observer?" + _3D_HINT + " Choose from the options.",
        "From the vantage point of {green_label} (green bbox) looking at {blue_label} (blue bbox), where would {red_label} (red bbox) be relative to you?" + _3D_HINT + " Select your answer.",
        "Standing at {green_label} (green bbox) facing {blue_label} (blue bbox), describe the direction of {red_label} (red bbox) from your perspective." + _3D_HINT + " Choose the correct response.",
        "Consider moving to {green_label} (green bbox) and orienting toward {blue_label} (blue bbox). In what direction is {red_label} (red bbox)?" + _3D_HINT + " Pick the right response.",
        "If the observer repositions to {green_label} (green bbox) while facing {blue_label} (blue bbox), where is {red_label} (red bbox) located?" + _3D_HINT + " Select one answer.",
        "From {green_label} (green bbox) facing {blue_label} (blue bbox), what is the spatial direction to {red_label} (red bbox)?" + _3D_HINT + " Choose the appropriate answer.",
    ],

    # ------------------------------------------------------------------
    # 016 compound_spatial_referring — sub_type "referring"
    # vars: target_label, rel_name, anchor_label
    # ------------------------------------------------------------------
    "compound_spatial_referring": [
        "Please point to the {target_label} {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Locate the {target_label} that is {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Identify the {target_label} {rel_name} the {anchor_label} in the image." + _COORD_SUFFIX,
        "Show the {target_label} positioned {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Point to the {target_label} {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Find the {target_label} that is {rel_name} the {anchor_label} and mark its location." + _COORD_SUFFIX,
        "Where is the {target_label} {rel_name} the {anchor_label}?" + _COORD_SUFFIX,
        "Indicate the position of the {target_label} {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Mark the {target_label} {rel_name} the {anchor_label}." + _COORD_SUFFIX,
        "Pinpoint the {target_label} {rel_name} the {anchor_label} in the image." + _COORD_SUFFIX,
    ],

    # ------------------------------------------------------------------
    # 016 compound_spatial_referring — sub_type "multi_step"
    # vars: move_to, face_toward, query
    # ------------------------------------------------------------------
    "compound_spatial_multi_step": [
        "Suppose I am at the {move_to} facing the {face_toward}. If I want to see the {query}, in which direction should I look?",
        "Imagine standing at the {move_to} and looking toward the {face_toward}. In which direction is the {query}?",
        "If I position myself at the {move_to} facing the {face_toward}, where should I look to find the {query}?",
        "Standing at the {move_to} and oriented toward the {face_toward}, which direction leads to the {query}?",
        "From the {move_to}, facing the {face_toward}, in what direction would I need to look to see the {query}?",
        "Assume I am at the {move_to} with my gaze toward the {face_toward}. Where is the {query} relative to me?",
        "If I stand at the {move_to} looking at the {face_toward}, which way should I turn to see the {query}?",
        "Positioned at the {move_to} and facing the {face_toward}, in which direction is the {query} located?",
        "At the {move_to}, looking toward the {face_toward}, which direction should I face to view the {query}?",
        "From the {move_to} with the {face_toward} ahead, in what direction would I find the {query}?",
    ],

    # ------------------------------------------------------------------
    # 017 object_placement
    # vars: label
    # ------------------------------------------------------------------
    "object_placement": [
        "Please point out the free space on the {label}." + _COORD_SUFFIX,
        "Locate an available area on the {label}." + _COORD_SUFFIX,
        "Identify an open space on the {label}." + _COORD_SUFFIX,
        "Where is there free space on the {label}?" + _COORD_SUFFIX,
        "Show an unoccupied area on the {label}." + _COORD_SUFFIX,
        "Point to an empty spot on the {label}." + _COORD_SUFFIX,
        "Find a free area on the {label} surface." + _COORD_SUFFIX,
        "Indicate where there is available space on the {label}." + _COORD_SUFFIX,
        "Mark a clear spot on the {label}." + _COORD_SUFFIX,
        "Pinpoint an open space on the {label}." + _COORD_SUFFIX,
    ],

    # ------------------------------------------------------------------
    # 018 spatial_compatibility
    # vars: moveable, dir_name, target
    # ------------------------------------------------------------------
    "spatial_compatibility": [
        "Can the {moveable} fit {dir_name} the {target}? Answer yes or no.",
        "Is there enough space for the {moveable} {dir_name} the {target}? Answer yes or no.",
        "Would the {moveable} fit if placed {dir_name} the {target}? Reply yes or no.",
        "Determine whether the {moveable} can be placed {dir_name} the {target}. Answer yes or no.",
        "Could the {moveable} be positioned {dir_name} the {target} without overlapping? Answer yes or no.",
        "Is it possible to fit the {moveable} {dir_name} the {target}? Respond with yes or no.",
        "Check if the {moveable} has room {dir_name} the {target}. Answer yes or no.",
        "Does the space {dir_name} the {target} accommodate the {moveable}? Answer yes or no.",
        "Can the {moveable} be accommodated {dir_name} the {target}? Reply yes or no.",
        "Is the {moveable} small enough to fit {dir_name} the {target}? Answer yes or no.",
    ],

    # ------------------------------------------------------------------
    # 019 object_matching_mv
    # vars: label, opts
    # ------------------------------------------------------------------
    "object_matching_mv": [
        "Based on the red bbox of {label} in the first image, locate its bounding box in the second image. Pick the appropriate answer from the options given.\n{opts}\nYour answer can only include one of the options A, B, C, or D.",
        "The {label} is marked with a red bbox in the first image. Which bounding box in the second image corresponds to the same object? Choose from:\n{opts}\nAnswer with A, B, C, or D only.",
        "Find the matching bounding box for {label} (red bbox in image 1) in the second image.\n{opts}\nSelect A, B, C, or D.",
        "The red bbox in the first image highlights the {label}. Identify the correct bounding box for the same object in the second image.\n{opts}\nPick one: A, B, C, or D.",
        "Locate the {label} shown by the red bbox in image 1 within the second image. Which option matches?\n{opts}\nAnswer A, B, C, or D.",
        "In the first image, {label} is bounded by a red box. Which bounding box in the second image shows the same object?\n{opts}\nChoose A, B, C, or D.",
        "Match the {label} (red bbox, first image) to its bounding box in the second image.\n{opts}\nRespond with A, B, C, or D only.",
        "The {label} is indicated by the red bbox in image 1. Select the corresponding bbox in image 2.\n{opts}\nAnswer with one letter: A, B, C, or D.",
        "Identify which bounding box in the second image matches the {label} (red bbox) from the first image.\n{opts}\nYour answer should be A, B, C, or D.",
        "Using the red bbox of {label} in the first image as reference, find its location in the second image.\n{opts}\nSelect A, B, C, or D.",
    ],

    # ------------------------------------------------------------------
    # 020 depth_difference
    # vars: label_a, label_b
    # ------------------------------------------------------------------
    "depth_difference": [
        "What is the depth difference between the {label_a} (blue point) and the {label_b} (red point)? Depth is measured along the camera's forward axis. The unit is meter. Answer using a single number and nothing else.",
        "How much do the depths of {label_a} (blue point) and {label_b} (red point) differ? Answer in meters with a single number.",
        "Calculate the difference in depth between {label_a} (blue point) and {label_b} (red point) in meters. Provide only a number.",
        "What is the depth separation in meters between {label_a} (blue point) and {label_b} (red point) along the viewing direction? Answer numerically.",
        "Determine the depth difference in meters between {label_a} (blue point) and {label_b} (red point). Reply with a single number.",
        "Measure the difference in camera-forward depth between {label_a} (blue point) and {label_b} (red point) in meters. Answer with one number.",
        "How many meters apart are {label_a} (blue point) and {label_b} (red point) in terms of depth? Provide a numerical answer.",
        "Report the depth difference between {label_a} (blue point) and {label_b} (red point) in meters. State a single number.",
        "What is the forward-axis depth gap between {label_a} (blue point) and {label_b} (red point)? Answer in meters, number only.",
        "Compute the depth difference in meters between {label_a} (blue point) and {label_b} (red point). Give only the numerical value.",
    ],

    # ------------------------------------------------------------------
    # 021 object_grounding_bbox
    # vars: label
    # ------------------------------------------------------------------
    "object_grounding_bbox": [
        "Please locate the {label} in the image. Your answer should be a bounding box [x_min, y_min, x_max, y_max] with normalized coordinates between 0 and 1.",
        "Find the {label} and provide its bounding box as [x_min, y_min, x_max, y_max] with coordinates normalized to 0-1.",
        "Where is the {label}? Answer with a bounding box [x_min, y_min, x_max, y_max] using normalized coordinates (0 to 1).",
        "Locate the {label} in the image and give its bounding box [x_min, y_min, x_max, y_max] in normalized coordinates.",
        "Identify the {label} and output its bounding box as [x_min, y_min, x_max, y_max] with values between 0 and 1.",
        "Detect the {label} in this image. Provide the bounding box [x_min, y_min, x_max, y_max] in normalized coordinates.",
        "Mark the {label} with a bounding box [x_min, y_min, x_max, y_max]. Use normalized coordinates (0 to 1).",
        "Point out the {label} by giving its bounding box [x_min, y_min, x_max, y_max] in normalized image coordinates.",
        "Specify the location of the {label} as a bounding box [x_min, y_min, x_max, y_max] with coordinates from 0 to 1.",
        "Provide the bounding box of the {label} in the format [x_min, y_min, x_max, y_max] with normalized coordinates.",
    ],

    # ------------------------------------------------------------------
    # 022 object_category (no vars)
    # ------------------------------------------------------------------
    "object_category": [
        "What category does the object marked with the red bounding box belong to?",
        "Identify the category of the object highlighted by the red bounding box.",
        "What type of object is enclosed by the red bounding box?",
        "Name the category of the object inside the red bounding box.",
        "What is the object within the red bounding box?",
        "Determine the category of the object indicated by the red bounding box.",
        "What kind of object is marked by the red bounding box?",
        "Classify the object shown inside the red bounding box.",
        "Which object category does the red bounding box highlight?",
        "What is the category of the item bounded by the red box?",
    ],

    # ------------------------------------------------------------------
    # 023 object_size_qualitative
    # vars: label
    # ------------------------------------------------------------------
    "object_size_qualitative": [
        "Classify the {label} (red bbox) by its longest dimension: small (<0.4 m), medium (0.4–1.2 m), or large (>1.2 m)?",
        "Based on longest dimension, is the {label} (red bbox) small (<0.4 m), medium (0.4–1.2 m), or large (>1.2 m)?",
        "Is the {label} (red bbox) small (<0.4 m), medium (0.4–1.2 m), or large (>1.2 m) by its longest dimension?",
        "Judge the size of the {label} (red bbox) by longest dimension: small (<0.4 m), medium (0.4–1.2 m), or large (>1.2 m)?",
        "Considering its longest dimension, is the {label} (red bbox) small (<0.4 m), medium (0.4–1.2 m), or large (>1.2 m)?",
    ],

    # ------------------------------------------------------------------
    # 024 comparative_spatial_grounding
    # vars: target_label, relation, anchor_label
    # ------------------------------------------------------------------
    "comparative_spatial_grounding": [
        "Point to the {target_label} {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Locate the {target_label} that is {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Identify the {target_label} {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Show the {target_label} {relation} the {anchor_label} in the image." + _COORD_SUFFIX,
        "Find the {target_label} {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Where is the {target_label} {relation} the {anchor_label}?" + _COORD_SUFFIX,
        "Indicate the {target_label} {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Mark the {target_label} that is {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Pinpoint the {target_label} {relation} the {anchor_label}." + _COORD_SUFFIX,
        "Point out the {target_label} {relation} the {anchor_label} in this image." + _COORD_SUFFIX,
    ],

    # ------------------------------------------------------------------
    # 026 object_object_position_mv
    # vars: label_a, label_b
    # ------------------------------------------------------------------
    "object_object_position_mv": [
        "In the two images, describe the location of the {label_a} (blue bbox) with respect to the {label_b} (red bbox), from the first camera's perspective. Select the correct response from the given choices.",
        "From the first camera's viewpoint, where is {label_a} (blue bbox) relative to {label_b} (red bbox) across the two images? Choose the correct answer.",
        "Considering the first camera's perspective, describe the spatial relationship of {label_a} (blue bbox) to {label_b} (red bbox) in the two images. Select one.",
        "Using the two images, describe where {label_a} (blue bbox) is positioned relative to {label_b} (red bbox) from the first camera's view. Pick the correct option.",
        "How is {label_a} (blue bbox) located with respect to {label_b} (red bbox) from the first camera's perspective? Select your answer.",
        "Across the two views, what is the position of {label_a} (blue bbox) relative to {label_b} (red bbox) as seen from the first camera? Choose from the choices.",
        "Describe the relative location of {label_a} (blue bbox) compared to {label_b} (red bbox) from the first camera angle. Select the correct response.",
        "From the perspective of the first camera, determine the spatial relationship between {label_a} (blue bbox) and {label_b} (red bbox). Pick the right answer.",
        "In these two images, how does {label_a} (blue bbox) relate spatially to {label_b} (red bbox) from the first camera's view? Choose the correct option.",
        "Based on the first camera's perspective, where is {label_a} (blue bbox) in relation to {label_b} (red bbox)? Select the appropriate answer.",
    ],

    # ------------------------------------------------------------------
    # 027 camera_region_position
    # vars: area_name
    # ------------------------------------------------------------------
    "camera_region_position": [
        "In which direction is the {area_name} relative to you?",
        "From your viewpoint, where is the {area_name}?",
        "Relative to the camera, in which direction is the {area_name}?",
        "Which direction is the {area_name} from the observer's position?",
        "From where you are standing, in which direction is the {area_name}?",
        "Looking from the camera's perspective, where is the {area_name}?",
        "In what direction would you find the {area_name} from the camera's position?",
        "Relative to the observer, which direction leads to the {area_name}?",
        "From the current viewpoint, where is the {area_name} located?",
        "Determine the direction of the {area_name} relative to the camera.",
    ],

    # ------------------------------------------------------------------
    # 028 object_region_position
    # vars: ref_dir, area_name, label
    # ------------------------------------------------------------------
    "object_region_position": [
        "Taking the camera's facing direction as {ref_dir}, in which direction is the {area_name} from the {label}?",
        "With the camera facing {ref_dir}, where is the {area_name} relative to the {label}?",
        "If the camera's forward direction is {ref_dir}, in what direction is the {area_name} from the {label}?",
        "Considering the camera faces {ref_dir}, describe the direction of the {area_name} from the {label}.",
        "Given the camera is oriented {ref_dir}, which direction is the {area_name} from the {label}?",
        "With {ref_dir} as the camera's facing direction, where is the {area_name} in relation to the {label}?",
        "The camera faces {ref_dir}. In which direction is the {area_name} from the {label}?",
        "Using {ref_dir} as the reference direction, where is the {area_name} relative to the {label}?",
        "Facing {ref_dir}, determine the direction of the {area_name} from the {label}.",
        "If forward is {ref_dir}, in which direction from the {label} is the {area_name}?",
    ],

    # ------------------------------------------------------------------
    # 029 region_region_position
    # vars: ref_dir, name_a, name_b
    # ------------------------------------------------------------------
    "region_region_position": [
        "Taking the camera's facing direction as {ref_dir}, in which direction is the {name_b} from the {name_a}?",
        "With the camera facing {ref_dir}, where is the {name_b} relative to the {name_a}?",
        "If the camera's forward direction is {ref_dir}, what direction is the {name_b} from the {name_a}?",
        "Considering the camera faces {ref_dir}, describe the direction of the {name_b} from the {name_a}.",
        "Given the camera is oriented {ref_dir}, which direction is the {name_b} from the {name_a}?",
        "With {ref_dir} as the reference direction, where is the {name_b} in relation to the {name_a}?",
        "The camera faces {ref_dir}. In which direction is the {name_b} from the {name_a}?",
        "Facing {ref_dir}, determine the direction of the {name_b} from the {name_a}.",
        "Using {ref_dir} as forward, in which direction from the {name_a} is the {name_b}?",
        "If forward is {ref_dir}, where is the {name_b} relative to the {name_a}?",
    ],

    # ------------------------------------------------------------------
    # 030 route_planning
    # vars: sl, fl, dl
    # ------------------------------------------------------------------
    "route_planning": [
        "You are a robot beginning at the center of the {sl} facing the {fl}. You want to navigate to the {dl}. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. [please fill in] 2. Go forward until the {dl}. You have reached the final destination.",
        "Starting at the {sl} and facing the {fl}, you need to reach the {dl}. Choose 'turn back,' 'turn left,' or 'turn right' for step 1, then go forward to the {dl}. 1. [please fill in] 2. Go forward until the {dl}.",
        "You begin at the {sl}, looking toward the {fl}. Navigate to the {dl}. For step 1, select 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Go forward until the {dl}.",
        "From the center of the {sl}, facing the {fl}, plan a route to the {dl}. Fill in step 1 with 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Proceed forward to the {dl}.",
        "A robot starts at the {sl} facing the {fl} and must reach the {dl}. What should step 1 be? Choose 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Go forward until the {dl}.",
        "Position: center of the {sl}, facing the {fl}. Goal: reach the {dl}. Pick 'turn back,' 'turn left,' or 'turn right' for step 1. 1. [please fill in] 2. Move forward to the {dl}.",
        "You are at the {sl} oriented toward the {fl}. To get to the {dl}, what action should you take first? Choose from 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Walk forward to the {dl}.",
        "Starting position: {sl}, facing: {fl}, destination: {dl}. What turn do you make first? Choose 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Go forward until the {dl}.",
        "Begin at the {sl} with the {fl} ahead. You want to reach the {dl}. Select your first action: 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Go forward to the {dl}.",
        "At the {sl} facing the {fl}, navigate to the {dl}. For step 1, choose 'turn back,' 'turn left,' or 'turn right.' 1. [please fill in] 2. Go forward until you reach the {dl}.",
    ],
}


def pick_template(
    qtype: str,
    rng: random.Random,
    **kwargs: str,
) -> str:
    """Pick a random template for *qtype* and format it with *kwargs*."""
    templates = TEMPLATES.get(qtype)
    if not templates:
        raise KeyError(f"No templates for question type {qtype!r}")
    tmpl = rng.choice(templates)
    return tmpl.format(**kwargs) if kwargs else tmpl
