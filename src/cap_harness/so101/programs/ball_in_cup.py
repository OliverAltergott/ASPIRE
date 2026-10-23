"""Ball-into-cup skill: one CAP program, no agent round trips.

See examples/so101/ball_in_cup/TASK_CONTEXT.md. Needs the SO101 runtime with --sam3-url;
run it with --program this file, or over MCP as cap_program_ball_in_cup. SAM3 locates the ball and the
mug once in the external view; the recipe's grasp (38 degree pitch, tool GRASP_PAST_MM past
the ball centre at z = 0) and placement (released at rim height) run as continuous moves.
Before closing, the wrist view must show the ball where successful closes had it; the
program corrects up to twice and otherwise stops without closing.

Placement maps the mug's pixel to a release tool position with a similarity fit over
releases that verifiably landed (RELEASES). A held ball cannot calibrate this: closing
pushes it 2-3 cm off the tool axis, and from the external camera the gripper hides its
inner part, which biased held-ball centroids by about 15 px. After a verified success,
the result's "release" pair can be appended to RELEASES.
"""

import math

REST = [-5.0, -106.3, 66.7, 97.2, -84.8]
PITCH = 38.0
SIN, COS = math.sin(math.radians(PITCH)), math.cos(math.radians(PITCH))
# Station data. TABLE: two pixel -> ball pairs (the ball 14 mm behind the grasping tool),
# reliable only near those pixels; the wrist check corrects grasp errors up to 25 mm.
# RELEASES: verified releases, (mug centre pixel, release tool x, y in mm) at PLACE_Z.
TABLE = ((370, 228, 219, 101), (386, 292, 276, 7))
RELEASES = (((346, 354), (237.0, -85.6)), ((384, 146), (249.3, 201.8)), ((382.5, 142), (236.5, 198.5)),
            ((365.2, 113.9), (212.7, 236.2)))
PLACE_Z = 102.0
# At 38 degrees the jaws' lower edge rises about 0.8 mm per mm behind the tips. With the
# ball centre 14 mm behind them, the closing jaw pushed the ball deeper and it rolled out
# under the jaws in two of four closes; 8 mm keeps the edge about 10 mm below its equator.
GRASP_PAST_MM = 8.0
# Wrist view (640x480) of the ball just before successful closes at 14 mm: centres
# (439, 307) and (417, 303), about 100 px across, so about 2.5 px/mm sideways; moving
# along the fingers shifted the ball about 2.7 px/mm vertically. 6 mm shallower moves
# it about 13 px up the image.
WRIST_TARGET, WRIST_TOLERANCE_PX = (428, 292), 12
LATERAL_PX_PER_MM, ALONG_PX_PER_MM = 2.5, 2.7


def two_point(pairs, pixel):
    # Similarity transform from two pairs; image v points down, so it is negated.
    (u1, v1, x1, y1), (u2, v2, x2, y2) = pairs
    du, dw, dx, dy = u2 - u1, v1 - v2, x2 - x1, y2 - y1
    norm = du * du + dw * dw
    real, imag = (dx * du + dy * dw) / norm, (dy * du - dx * dw) / norm
    pu, pw = pixel[0] - u1, v1 - pixel[1]
    return [x1 + real * pu - imag * pw, y1 + imag * pu + real * pw]


def tool_for(xy, z, offset_mm=14.0):
    heading = math.atan2(xy[1], xy[0])
    return [xy[0] + offset_mm * math.cos(heading), xy[1] + offset_mm * math.sin(heading), z]


def release_tool_xy(pixel):
    # Least-squares similarity (scale, rotation, shift) from RELEASES; image v points down.
    count = len(RELEASES)
    mu = sum(r[0][0] for r in RELEASES) / count
    mw = -sum(r[0][1] for r in RELEASES) / count
    mx = sum(r[1][0] for r in RELEASES) / count
    my = sum(r[1][1] for r in RELEASES) / count
    norm = real = imag = 0.0
    for (u, v), (x, y) in RELEASES:
        du, dw, dx, dy = u - mu, -v - mw, x - mx, y - my
        norm += du * du + dw * dw
        real += du * dx + dw * dy
        imag += du * dy - dw * dx
    real, imag = real / norm, imag / norm
    du, dw = pixel[0] - mu, -pixel[1] - mw
    return [mx + real * du - imag * dw, my + imag * du + real * dw]


def check(step, outcome):
    if not outcome.ok:
        raise ValueError(f"{step} failed: {outcome.error}")
    return outcome


def tool_now():
    return so101.get_telemetry()["urdf_model"]["goal_tool_xyz_mm"]


def move_to(target):
    here = tool_now()
    return so101.move_tool([target[i] - here[i] for i in range(3)])


def stage_behind(grasp):
    # Furthest reachable standoff along the finger axis, so the approach slides along the fingers.
    heading = math.atan2(grasp[1], grasp[0])
    for back in (90, 75, 60, 45, 35):
        staged = so101.go_to(xyz_mm=[grasp[0] - back * COS * math.cos(heading),
                                     grasp[1] - back * COS * math.sin(heading), grasp[2] + back * SIN], pitch=PITCH)
        if staged.ok:
            return back
        if "unreachable" not in str(staged.error):
            raise ValueError(f"stage failed: {staged.error}")
    raise ValueError("No reachable staging pose behind the ball; it is too close to the base")


wrist_log = []


def centre_ball_between_jaws():
    corrections = []
    for attempt in range(3):
        seen = so101.segment_text("green ball", "wrist")["wrist"]
        if not seen:
            raise ValueError("The ball is not visible between the jaws; not closing")
        wrist_log.append({"center_px": seen[0]["center_px"], "box_xyxy": seen[0]["box_xyxy"]})
        dx = seen[0]["center_px"][0] - WRIST_TARGET[0]
        dy = seen[0]["center_px"][1] - WRIST_TARGET[1]
        if abs(dx) <= WRIST_TOLERANCE_PX and abs(dy) <= WRIST_TOLERANCE_PX:
            return corrections
        if attempt == 2:
            raise ValueError(f"The ball stayed {dx:.0f}, {dy:.0f} px from the grasp window; not closing")
        lateral = max(-25.0, min(25.0, dx / LATERAL_PX_PER_MM))  # + is image right: clockwise
        along = max(-20.0, min(20.0, -dy / ALONG_PX_PER_MM))  # + is further along the fingers
        here = tool_now()
        heading = math.atan2(here[1], here[0])
        corrections.append([round(lateral, 1), round(along, 1)])
        check("lift", so101.move_tool([0, 0, 8]))
        check("centre", so101.move_tool([lateral * math.sin(heading) + along * COS * math.cos(heading),
                                         -lateral * math.cos(heading) + along * COS * math.sin(heading),
                                         -along * SIN]))
        check("lower", so101.move_tool([0, 0, -8]))
    return corrections


obs = get_observation()
check("enable", so101.enable())
check("rest", so101.go_to(joints_deg=REST))  # A clear external view; near-instant when already at rest.
seen = so101.segment_text(["green ball", "mug"], "external")
ball_seen, mug_seen = seen["green ball"]["external"], seen["mug"]["external"]
if not ball_seen or not mug_seen:
    raise ValueError(f"SAM3 did not find both objects: ball={ball_seen}, mug={mug_seen}")
ball = two_point(TABLE, ball_seen[0]["center_px"])

grasp = tool_for(ball, 0.0, GRASP_PAST_MM)
standoff = stage_behind(grasp)
check("open", set_gripper(0.55))
check("approach", move_to(grasp))
corrections = centre_ball_between_jaws()
contact = so101.close_until_contact(0.1)
if not contact.ok:
    raise ValueError(f"grasp failed: {contact.error}; wrist checks {wrist_log}")
check("lift", so101.move_tool([0, 0, 70]))
if so101.get_telemetry()["positions"]["gripper"] < 19:
    raise ValueError("The gripper closed further after lifting: the ball was not retained")

place_xy = release_tool_xy(mug_seen[0]["center_px"])
place = [place_xy[0], place_xy[1], PLACE_Z]
here = tool_now()
check("carry", move_to([(here[0] + place[0]) / 2, (here[1] + place[1]) / 2, 130.0]))
check("carry", move_to([place[0], place[1], 130.0]))
check("lower", move_to(place))
check("release", set_gripper(0.48))
check("retreat", so101.move_tool([0, 0, 30]))
check("rest", so101.go_to(joints_deg=REST))

after = so101.segment_text(["green ball", "mug"], "external")
ball_after, mug_after = after["green ball"]["external"], after["mug"]["external"] or mug_seen
x0, y0, x1, y1 = mug_after[0]["box_xyxy"]
inside = bool(ball_after) and x0 <= ball_after[0]["center_px"][0] <= x1 and y0 <= ball_after[0]["center_px"][1] <= y1
result = {"standoff_mm": standoff, "ball_mm": [round(v) for v in ball], "place_tool_mm": [round(v) for v in place],
          "wrist_corrections_mm": corrections, "wrist_checks": wrist_log, "contact_opening": contact.diagnostics["opening"],
          "ball_in_mug_box": inside, "ball_seen_after_px": ball_after[0]["center_px"] if ball_after else None,
          "release": [mug_seen[0]["center_px"], [round(v, 1) for v in place_xy]]}
print(result)
if ball_after and not inside:
    raise ValueError(f"The ball ended outside the mug, at external pixel {ball_after[0]['center_px']}: {result}")
