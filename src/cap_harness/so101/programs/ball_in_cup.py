"""Ball-into-cup skill: one CAP program, no agent round trips (see TASK_CONTEXT.md).

Run it through the SO101 runtime started with --sam3-url, e.g. --program this file.
It locates the ball and the mug once with SAM3 in the external view, then executes the
recipe's grasp (38 degree pitch, tool 14 mm past the ball centre at z = 0) and placement
(ball centre 14 mm behind the tool, released at rim height) as continuous moves.

Station data: the pixel-to-millimetre pairs below were measured in earlier runs on this
station and are reliable only near those pixels; recalibrate after moving the external
camera or placing objects elsewhere.
"""

import math

REST = [-5.0, -106.3, 66.7, 97.2, -84.8]
PITCH = 38.0
TABLE = ((370, 228, 219, 101), (386, 292, 276, 7))  # pixel -> mm at ball-centre height
RIM = ((346, 354, 223, -80), (384, 146, 238, 193))  # pixel -> mm at mug-rim height


def to_robot(pairs, pixel):
    # Similarity transform from two pairs; image v points down, so it is negated.
    (u1, v1, x1, y1), (u2, v2, x2, y2) = pairs
    du, dw, dx, dy = u2 - u1, v1 - v2, x2 - x1, y2 - y1
    norm = du * du + dw * dw
    real, imag = (dx * du + dy * dw) / norm, (dy * du - dx * dw) / norm
    pu, pw = pixel[0] - u1, v1 - pixel[1]
    return [x1 + real * pu - imag * pw, y1 + imag * pu + real * pw]


def detect(query):
    found = so101.segment_text(query, "external")["external"]
    return found[0] if found else None


def tool_for(xy, z, offset_mm=14.0):
    heading = math.atan2(xy[1], xy[0])
    return [xy[0] + offset_mm * math.cos(heading), xy[1] + offset_mm * math.sin(heading), z]


def check(step, outcome):
    if not outcome.ok:
        raise ValueError(f"{step} failed: {outcome.error}")
    return outcome


def move_to(target):
    here = so101.get_telemetry()["urdf_model"]["goal_tool_xyz_mm"]
    return so101.move_tool([target[i] - here[i] for i in range(3)])


obs = get_observation()
check("enable", so101.enable())
ball_px, mug_px = detect("green ball"), detect("mug")
if ball_px is None or mug_px is None:
    raise ValueError(f"SAM3 did not find both objects: ball={ball_px}, mug={mug_px}")
ball, mug = to_robot(TABLE, ball_px["center_px"]), to_robot(RIM, mug_px["center_px"])

grasp = tool_for(ball, 0.0)
heading = math.atan2(grasp[1], grasp[0])
check("stage", so101.go_to(xyz_mm=[grasp[0] - 70 * math.cos(heading), grasp[1] - 70 * math.sin(heading), 51.0],
                           pitch=PITCH))
check("open", set_gripper(0.55))
check("approach", move_to(grasp))
contact = check("grasp", so101.close_until_contact(0.1))
check("lift", so101.move_tool([0, 0, 70]))
if so101.get_telemetry()["positions"]["gripper"] < 19:
    raise ValueError("The gripper closed further after lifting: the ball was not retained")

place = tool_for(mug, 102.0)
lifted = so101.get_telemetry()["urdf_model"]["goal_tool_xyz_mm"]
check("carry", move_to([(lifted[0] + place[0]) / 2, (lifted[1] + place[1]) / 2, 130.0]))
check("carry", move_to([place[0], place[1], 130.0]))
check("lower", move_to(place))
check("release", set_gripper(0.48))
check("retreat", so101.move_tool([0, 0, 30]))
check("rest", so101.go_to(joints_deg=REST))

after = detect("green ball")
x0, y0, x1, y1 = mug_px["box_xyxy"]
inside = after is not None and x0 <= after["center_px"][0] <= x1 and y0 <= after["center_px"][1] <= y1
result = {"ball_mm": [round(v) for v in ball], "mug_mm": [round(v) for v in mug],
          "contact_opening": contact.diagnostics["opening"],
          "ball_after_px": None if after is None else after["center_px"], "ball_in_mug_box": inside}
print(result)
