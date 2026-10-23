"""Ball-into-cup skill: one CAP program, no agent round trips.

See examples/so101/ball_in_cup/TASK_CONTEXT.md. Needs the SO101 runtime with --sam3-url
(and, once recorded, --overhead-calibration); run it with --program this file, or over
MCP as cap_program_ball_in_cup. SAM3 locates the ball and the
mug once in the external view; the recipe's grasp (38 degree pitch, tool 14 mm past the
ball centre at z = 0) and placement (released at rim height) run as continuous moves.
Before closing, the wrist view must show the ball where successful closes had it; the
program corrects up to twice and otherwise stops without closing.

With CALIBRATE_OVERHEAD = True the held ball also visits six poses per plane and the
result carries "calibration_records"; fit them with
`python -m cap_harness.so101.calibration result.json overhead.json`.
"""

import math

REST = [-5.0, -106.3, 66.7, 97.2, -84.8]
PITCH = 38.0
SIN, COS = math.sin(math.radians(PITCH)), math.cos(math.radians(PITCH))
# Fallback station data when no calibration is loaded: two pixel -> mm pairs per height
# from earlier runs, reliable only near those pixels.
TABLE = ((370, 228, 219, 101), (386, 292, 276, 7))
RIM = ((346, 354, 223, -80), (384, 146, 238, 193))
# Wrist view (640x480) of the ball just before the two successful closes: centres
# (439, 307) and (417, 303), about 100 px across, so about 2.5 px/mm sideways; moving
# along the fingers shifted the ball about 2.7 px/mm vertically.
WRIST_TARGET, WRIST_TOLERANCE_PX = (428, 305), 12
LATERAL_PX_PER_MM, ALONG_PX_PER_MM = 2.5, 2.7
CALIBRATE_OVERHEAD = False
# Tool poses (z mm, [(radius mm, heading deg)]) for the held ball, at least 100 mm from the mug.
CALIBRATION_POSES = (
    ("table", 10.0, ((245, -25), (300, -25), (245, 0), (305, 0), (245, 22), (300, 18))),
    ("rim", 102.0, ((300, 20), (300, -15), (240, -15), (240, 20), (240, 60), (300, 75))),
)


def two_point(pairs, pixel):
    # Similarity transform from two pairs; image v points down, so it is negated.
    (u1, v1, x1, y1), (u2, v2, x2, y2) = pairs
    du, dw, dx, dy = u2 - u1, v1 - v2, x2 - x1, y2 - y1
    norm = du * du + dw * dw
    real, imag = (dx * du + dy * dw) / norm, (dy * du - dx * dw) / norm
    pu, pw = pixel[0] - u1, v1 - pixel[1]
    return [x1 + real * pu - imag * pw, y1 + imag * pu + real * pw]


def to_base(plane, pixel):
    try:
        return so101.pixel_to_base(pixel, plane), "calibration"
    except Exception:
        return two_point(TABLE if plane == "table" else RIM, pixel), "two-point fallback"


def held_ball_mm():
    model = so101.get_telemetry()["urdf_model"]
    x, y = model["tool_xyz_mm"][0], model["tool_xyz_mm"][1]
    pitch, heading = math.radians(model["pitch_deg"]), math.atan2(y, x)
    return [x - 18 * math.cos(pitch) * math.cos(heading), y - 18 * math.cos(pitch) * math.sin(heading)]


def affine_map(pairs):
    """Least-squares affine pixel -> mm from (u, v, x, y) pairs; None if degenerate."""
    normal = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    rhs = [[0.0] * 3, [0.0] * 3]
    for u, v, x, y in pairs:
        row = (u, v, 1.0)
        for i in range(3):
            rhs[0][i] += row[i] * x
            rhs[1][i] += row[i] * y
            for j in range(3):
                normal[i][j] += row[i] * row[j]

    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    scale = det(normal)
    if abs(scale) < 1e-6:
        return None
    coefficients = [[det([[b[r] if c == k else normal[r][c] for c in range(3)] for r in range(3)]) / scale
                     for k in range(3)] for b in rhs]
    return lambda pixel: [c[0] * pixel[0] + c[1] * pixel[1] + c[2] for c in coefficients]


def tool_for(xy, z, offset_mm=14.0):
    heading = math.atan2(xy[1], xy[0])
    return [xy[0] + offset_mm * math.cos(heading), xy[1] + offset_mm * math.sin(heading), z]


def check(step, outcome):
    if not outcome.ok:
        raise ValueError(f"{step} failed: {outcome.error}")
    return outcome


def tool_now():
    return so101.get_telemetry()["urdf_model"]["goal_tool_xyz_mm"]


def move_to(target):
    here = tool_now()
    return so101.move_tool([target[i] - here[i] for i in range(3)])


def centre_ball_between_jaws():
    corrections = []
    for attempt in range(3):
        seen = so101.segment_text("green ball", "wrist")["wrist"]
        if not seen:
            raise ValueError("The ball is not visible between the jaws; not closing")
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
ball, mapping = to_base("table", ball_seen[0]["center_px"])
mug, mug_mapping = to_base("rim", mug_seen[0]["center_px"])

grasp = tool_for(ball, 0.0)
heading = math.atan2(grasp[1], grasp[0])
check("stage", so101.go_to(xyz_mm=[grasp[0] - 70 * math.cos(heading), grasp[1] - 70 * math.sin(heading), 51.0],
                           pitch=PITCH))
check("open", set_gripper(0.55))
check("approach", move_to(grasp))
corrections = centre_ball_between_jaws()
contact = check("grasp", so101.close_until_contact(0.1))
check("lift", so101.move_tool([0, 0, 70]))
if so101.get_telemetry()["positions"]["gripper"] < 19:
    raise ValueError("The gripper closed further after lifting: the ball was not retained")

records = []
if CALIBRATE_OVERHEAD:
    for plane, z, poses in CALIBRATION_POSES:
        for radius, degrees in poses:
            check("calibrate", move_to([radius * math.cos(math.radians(degrees)),
                                        radius * math.sin(math.radians(degrees)), z]))
            held = so101.segment_text("green ball", "external")["external"]
            if held:
                positions = so101.get_telemetry()["positions"]
                records.append({"plane": plane, "pixel": held[0]["center_px"], "ball_mm": held_ball_mm(),
                                "joints_deg": [positions[j] for j in
                                               ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]})
    here = tool_now()
    check("clear", move_to([here[0], here[1], 130.0]))
    # The rim-height records just taken locate the mug better than the fallback pairs.
    rim_map = affine_map([(r["pixel"][0], r["pixel"][1], r["ball_mm"][0], r["ball_mm"][1])
                          for r in records if r["plane"] == "rim"])
    if rim_map is not None:
        mug, mug_mapping = rim_map(mug_seen[0]["center_px"]), "in-run affine"

place = tool_for(mug, 102.0)
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
result = {"map": mapping, "mug_map": mug_mapping, "ball_mm": [round(v) for v in ball], "mug_mm": [round(v) for v in mug],
          "wrist_corrections_mm": corrections, "contact_opening": contact.diagnostics["opening"],
          "ball_in_mug_box": inside, "calibration_records": records}
print({key: value for key, value in result.items() if key != "calibration_records"})
