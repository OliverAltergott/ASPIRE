from copy import deepcopy
import time

import numpy as np
import pytest

from cap_harness.contracts import Segmentation
from cap_harness.geometry import mask_to_point_cloud
from cap_harness.runtime import ProgramExecutor
from cap_harness.so101.adapter import JOINTS, MOTORS, So101Adapter
from cap_harness.so101.kinematics import (
    TOOL_PATH_STEP_DEG,
    pitch_deg,
    plan_tool_path,
    tool_transform,
    tool_xyz_mm,
)


class Hardware:
    def __init__(self):
        self.values = {
            "positions": dict.fromkeys(MOTORS, 10.0),
            "goals": dict.fromkeys(MOTORS, -170.0),
            "mode": dict.fromkeys(MOTORS, 0), "torque": dict.fromkeys(MOTORS, 0),
            "limits": {j: [0, 100] if j == "gripper" else [-120, 120] for j in MOTORS},
            "loads": dict.fromkeys(MOTORS, 0), "calibrated": True,
        }
        self.writes = []
        self.camera_error = False
        self.track = True
        self.closed = False

    def read(self):
        return {**deepcopy(self.values), "timestamp_s": time.time()}

    def frames(self):
        if self.camera_error:
            raise TimeoutError("camera unavailable")
        return {name: (np.zeros((3, 4, 3), dtype=np.uint8), time.time()) for name in ("external", "wrist")}

    def write(self, goals):
        self.writes.append(("goals", dict(goals)))
        self.values["goals"].update(goals)
        if self.track:
            self.values["positions"].update(goals)

    def enable(self):
        self.writes.append(("enable", None))
        self.values["torque"] = dict.fromkeys(MOTORS, 1)

    def close(self):
        self.closed = True


def ready():
    hw = Hardware()
    adapter = So101Adapter(hw, allow_motion=True, sleep=lambda _seconds: None)
    adapter.get_observation()
    assert adapter.enable().ok
    hw.writes.clear()
    return adapter, hw


def test_readonly_and_no_implicit_home_or_configuration():
    hw = Hardware()
    adapter = So101Adapter(hw)
    obs = adapter.get_observation()
    assert obs.robot_state.joint_positions["primary"].shape == (5,)
    np.testing.assert_allclose(obs.robot_state.joint_positions["primary"], np.deg2rad(10))
    assert obs.robot_state.gripper_positions["primary"] == 0.1
    assert not obs.robot_state.end_effector_poses
    assert not adapter.enable().ok
    assert not adapter.set_gripper(0.2).ok
    adapter.close()
    assert not hw.writes


def test_prime_pose_before_enabling_and_preserve_gains_and_untouched_goals():
    adapter, hw = ready()
    hw.values["positions"]["shoulder_lift"] -= 1
    assert adapter.move_delta({"shoulder_pan": 2}).ok
    assert all(set(goals) == {"shoulder_pan"} for _, goals in hw.writes)
    assert hw.values["goals"]["shoulder_lift"] == 10
    assert hw.values["goals"]["shoulder_pan"] == 12


def test_enable_primes_all_goals_before_torque():
    hw = Hardware()
    adapter = So101Adapter(hw, allow_motion=True)
    adapter.get_observation()
    assert adapter.enable().ok
    assert hw.writes == [("goals", dict.fromkeys(MOTORS, 10)), ("enable", None)]


@pytest.mark.parametrize("delta", [6, float("nan"), float("inf"), 200])
def test_bad_target_never_writes(delta):
    adapter, hw = ready()
    assert not adapter.move_delta({"shoulder_pan": delta}).ok
    assert not hw.writes


def test_gripper_units_are_separate_from_five_arm_radians():
    adapter, hw = ready()
    assert adapter.set_gripper(0.2).ok
    assert hw.values["goals"]["gripper"] == 20
    assert all(set(goals) == {"gripper"} for _, goals in hw.writes)
    assert not adapter.move_to_joints([0] * 6).ok
    hw.writes.clear()
    assert adapter.set_gripper(0.8).ok
    goals = [20] + [g["gripper"] for _, g in hw.writes]
    assert goals[-1] == pytest.approx(80) and max(abs(np.diff(goals))) <= 10 + 1e-9


def test_camera_failure_and_external_goal_changes_prevent_writes():
    adapter, hw = ready()
    hw.camera_error = True
    assert not adapter.move_delta({"shoulder_pan": 1}).ok
    assert not hw.writes
    hw.camera_error = False
    hw.values["goals"]["elbow_flex"] += 1
    assert not adapter.move_delta({"shoulder_pan": 1}).ok
    assert not hw.writes


def test_incomplete_tracking_is_failure_not_grasp_success():
    adapter, hw = ready()
    hw.track = False
    result = adapter.set_gripper(0.2)
    assert not result.ok
    assert result.final_observation is not None
    assert result.diagnostics["target_errors_degrees_percent"]["gripper"] == 10
    # Holding the requested gripper goal must survive an arm move.
    hw.track = True
    assert adapter.move_delta({"shoulder_pan": 1}).ok
    assert hw.values["goals"]["gripper"] == 20


def test_rgb_has_no_fabricated_depth_or_pose_and_projection_refuses():
    adapter, _ = ready()
    camera = adapter.get_observation().cameras["external"]
    assert camera.depth_m is camera.intrinsics is camera.camera_pose is None
    mask = Segmentation(mask=np.ones((3, 4), dtype=bool), label="ball", score=1,
                        frame=camera.frame, camera_name="external")
    with pytest.raises(ValueError, match="no calibrated metric depth"):
        mask_to_point_cloud(camera, mask)


def test_executor_uses_registered_tools_and_records_motion():
    adapter, hw = ready()
    result = ProgramExecutor(adapter.register_tools()).execute_program(
        "result = so101.move_delta({'shoulder_pan': 1})")
    assert result.ok and result.result.ok
    assert result.calls[0].tool == "so101.move_delta"
    assert hw.values["goals"][JOINTS[0]] == 11
    assert "move_to_pose" not in adapter.register_tools().public_tools()


REAL_POSE = {"shoulder_pan": -25.9, "shoulder_lift": -60.2, "elbow_flex": 73.2,
             "wrist_flex": 25.8, "wrist_roll": -84.8, "gripper": 50.0}


def ready_at(hw):
    hw.values["positions"].update(REAL_POSE)
    adapter = So101Adapter(hw, allow_motion=True, sleep=lambda _seconds: None)
    adapter.get_observation()
    assert adapter.enable().ok
    hw.writes.clear()
    return adapter


def arm_goals(hw):
    return [hw.values["goals"][j] for j in JOINTS]


def test_urdf_tool_point_and_pitch_convention():
    np.testing.assert_allclose(tool_xyz_mm([REAL_POSE[j] for j in JOINTS]), [218.0, 95.6, 71.4], atol=0.1)
    down = tool_transform([0, -20, 60, 50, 0])
    np.testing.assert_allclose(down[:3, 2], [0, 0, -1], atol=1e-6)


def test_tool_path_is_straight_bounded_and_respects_floor():
    start = [REAL_POSE[j] for j in JOINTS]
    limits = [[-120, 120]] * 5
    path = plan_tool_path(start, [20, 0, -30], None, limits, floor_z_mm=-8)
    np.testing.assert_allclose(tool_xyz_mm(path[-1]), tool_xyz_mm(start) + [20, 0, -30], atol=0.5)
    assert np.abs(np.diff(np.vstack([start, *path]), axis=0)).max() <= TOOL_PATH_STEP_DEG + 1e-9
    assert abs(pitch_deg(path[-1]) - pitch_deg(start)) < 0.2
    with pytest.raises(ValueError, match="table floor"):
        plan_tool_path(start, [0, 0, -100], None, limits, floor_z_mm=-8)


def test_move_tool_moves_tool_point_and_refuses_below_floor_without_writes():
    hw = Hardware()
    adapter = ready_at(hw)
    start = tool_xyz_mm(arm_goals(hw))
    result = adapter.move_tool([0, 10, 25], pitch=60)
    assert result.ok, result.error
    np.testing.assert_allclose(tool_xyz_mm(arm_goals(hw)), start + [0, 10, 25], atol=0.5)
    assert abs(pitch_deg(arm_goals(hw)) - 60) < 0.2
    assert result.diagnostics["tool_xyz_mm"] == pytest.approx(result.diagnostics["target_tool_xyz_mm"], abs=0.5)
    hw.writes.clear()
    assert not adapter.move_tool([0, 0, -150]).ok
    assert not adapter.move_tool([0, 0, float("nan")]).ok
    assert not hw.writes


class StallingHardware(Hardware):
    """Gripper jaws stop on an object at opening 20."""

    def write(self, goals):
        super().write(goals)
        if "gripper" in goals:
            self.values["positions"]["gripper"] = max(goals["gripper"], 20.0)
            self.values["loads"]["gripper"] = 500 if goals["gripper"] < 20 else 0


def test_close_until_contact_reports_stall_and_eases_squeeze():
    hw = StallingHardware()
    adapter = ready_at(hw)
    result = adapter.close_until_contact(0.05)
    assert result.ok
    assert result.diagnostics["contact"] and result.diagnostics["opening"] == 0.2
    assert hw.values["goals"]["gripper"] == pytest.approx(17.5)


def test_close_until_contact_without_object_is_not_success():
    hw = Hardware()
    adapter = ready_at(hw)
    result = adapter.close_until_contact(0.1)
    assert not result.ok and result.diagnostics["contact"] is False
    assert hw.values["goals"]["gripper"] == pytest.approx(10)


def test_detect_color_finds_disc_center_and_radius():
    cv2 = pytest.importorskip("cv2")

    class Disc(Hardware):
        def frames(self):
            rgb = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.circle(rgb, (100, 40), 15, (0, 255, 0), -1)
            return {name: (rgb.copy(), time.time()) for name in ("external", "wrist")}

    adapter = So101Adapter(Disc())
    blobs = adapter.detect_color([50, 100, 100], [70, 255, 255], "wrist")
    assert list(blobs) == ["wrist"]
    (blob,) = blobs["wrist"]
    assert blob["center_px"] == pytest.approx([100, 40], abs=1)
    assert blob["radius_px"] == pytest.approx(15, abs=1.5)


def test_go_to_joint_and_tool_targets_and_floor_refusal():
    hw = Hardware()
    adapter = ready_at(hw)
    rest = [-5.0, -100.0, 66.8, 97.5, -84.8]
    result = adapter.go_to(joints_deg=rest)
    assert result.ok, result.error
    np.testing.assert_allclose(arm_goals(hw), rest, atol=1e-6)
    steps = np.vstack([[REAL_POSE[j] for j in JOINTS]] + [[g.get(j, np.nan) for j in JOINTS] for _, g in hw.writes])
    assert np.nanmax(np.abs(np.diff(steps, axis=0))) <= TOOL_PATH_STEP_DEG + 1e-6

    target = tool_xyz_mm([REAL_POSE[j] for j in JOINTS]) + [20, 0, -10]
    result = adapter.go_to(xyz_mm=target, pitch=50)
    assert result.ok, result.error
    np.testing.assert_allclose(tool_xyz_mm(arm_goals(hw)), target, atol=0.5)
    assert abs(pitch_deg(arm_goals(hw)) - 50) < 0.2

    low = [REAL_POSE[j] for j in JOINTS]
    low[1] = 40.0
    assert tool_xyz_mm(low)[2] < -8
    hw.writes.clear()
    assert not adapter.go_to(joints_deg=low).ok
    assert not adapter.go_to(xyz_mm=[0, 0, 900], pitch=40).ok
    assert not hw.writes


class FakeSegmentation:
    def segment_text(self, observation, query):
        from cap_harness.contracts import Segmentation, SegmentationSet

        mask = np.zeros(observation.rgb.shape[:2], dtype=bool)
        mask[1:3, 2:4] = True
        return SegmentationSet(ok=True, segmentations=[Segmentation(
            mask=mask, label=query, score=0.9, camera_name=observation.frame, frame=observation.frame,
            box_xyxy=np.array([2.0, 1.0, 3.0, 2.0]))])


def test_segment_text_reports_box_and_mask_centroid_per_camera():
    adapter = So101Adapter(Hardware(), segmentation_provider=FakeSegmentation())
    assert "so101.segment_text" in adapter.register_tools().public_tools()
    detections = adapter.segment_text("ball", "wrist")
    assert detections == {"wrist": [{"score": 0.9, "box_xyxy": [2.0, 1.0, 3.0, 2.0],
                                      "center_px": [2.5, 1.5], "area_px": 4}]}
    assert "so101.segment_text" not in So101Adapter(Hardware()).register_tools().public_tools()


def test_mcp_bridge_serves_so101_tools_with_one_image_per_observe():
    pytest.importorskip("mcp")
    from mcp.types import ImageContent

    from cap_harness.so101.system2 import So101McpBridge, create_so101_app

    hw = Hardware()
    adapter = So101Adapter(hw, allow_motion=True, sleep=lambda _seconds: None)
    _app, session = create_so101_app(adapter)
    try:
        bridge = So101McpBridge(session)
        names = {spec.name for spec in bridge.tool_specs()}
        assert {"observe", "cap_so101__move_tool", "cap_so101__go_to", "cap_set_gripper"} <= names
        assert [name for name in names if name.startswith("cap_program_")] == ["cap_program_ball_in_cup"]
        assert "cap_so101__segment_text" not in names

        def images(result):
            return [item for item in result.content if isinstance(item, ImageContent)]

        observed = bridge.call("observe", {})
        assert not observed.isError and len(images(observed)) == 1
        enabled = bridge.call("cap_so101__enable", {})
        assert not enabled.isError, enabled.content[0].text
        moved = bridge.call("cap_so101__move_delta", {"deltas_deg": {"shoulder_pan": 1}})
        assert not moved.isError and not images(moved)
        assert hw.values["goals"]["shoulder_pan"] == 11
    finally:
        session.close()


REST_POSE = [-5.0, -106.3, 66.7, 97.2, -84.8]


class SceneHardware(StallingHardware):
    """Stalls the jaws on the ball and remembers the arm pose at which it was released."""

    def __init__(self):
        super().__init__()
        self.opened = self.held = False
        self.release_pose = None

    def frames(self):
        return {name: (np.zeros((480, 640, 3), dtype=np.uint8), time.time()) for name in ("external", "wrist")}

    def write(self, goals):
        super().write(goals)
        if "gripper" in goals:
            if goals["gripper"] > 40 and not self.held:
                self.opened = True
            elif self.opened and goals["gripper"] < 20:
                self.held = True
            elif self.held and goals["gripper"] > 40:
                self.held, self.release_pose = False, [self.values["goals"][j] for j in JOINTS]


class SceneSegmentation:
    """Fake SAM3: the resting ball and the mug where the real scene had them; after release
    the ball is reported inside the mug unless after_release says otherwise. Successive
    wrist views report the given ball centres, repeating the last.
    """

    def __init__(self, wrist_balls=((428, 292),)):
        self.centres = {("external", "green ball"): (385, 290), ("external", "mug"): (382, 142)}
        self.wrist_balls = list(wrist_balls)
        self.hardware = None
        self.after_release = None
        self.calls = []

    def segment_text(self, observation, query):
        from cap_harness.contracts import Segmentation, SegmentationSet

        camera = observation.frame.removesuffix("_optical")
        self.calls.append((camera, query))
        if camera == "wrist":
            u, v = self.wrist_balls.pop(0) if len(self.wrist_balls) > 1 else self.wrist_balls[0]
        elif query == "green ball" and self.hardware.release_pose is not None:
            u, v = self.after_release or self.centres[("external", "mug")]
        else:
            u, v = self.centres[(camera, query)]
        mask = np.zeros(observation.rgb.shape[:2], dtype=bool)
        mask[v - 5:v + 6, u - 5:u + 6] = True
        return SegmentationSet(ok=True, segmentations=[Segmentation(
            mask=mask, label=query, score=0.9, camera_name=observation.frame, frame=observation.frame,
            box_xyxy=np.array([u - 30.0, v - 30.0, u + 30.0, v + 30.0]))])


def run_skill(wrist_balls=((428, 292),), segmentation=None):
    from pathlib import Path

    hw = SceneHardware()
    hw.values["positions"].update(dict(zip(JOINTS, REST_POSE, strict=True)), gripper=2.0)
    segmentation = segmentation or SceneSegmentation(wrist_balls)
    segmentation.hardware = hw
    adapter = So101Adapter(hw, allow_motion=True, sleep=lambda _seconds: None, segmentation_provider=segmentation,
                           speed_deg_s=60.0)
    source = (Path(__file__).parents[1] / "src/cap_harness/so101/programs/ball_in_cup.py").read_text()
    return ProgramExecutor(adapter.register_tools()).execute_program(source), hw, segmentation


def test_ball_in_cup_skill_runs_end_to_end_with_fallback_maps():
    outcome, hw, segmentation = run_skill()
    assert outcome.ok, outcome.error
    result = outcome.result
    assert result["wrist_corrections_mm"] == ()
    assert result["contact_opening"] == pytest.approx(0.2) and result["ball_in_mug_box"]
    assert ("wrist", "green ball") in segmentation.calls
    np.testing.assert_allclose(arm_goals(hw), REST_POSE, atol=1e-6)
    assert hw.values["goals"]["gripper"] == pytest.approx(48)


def test_ball_in_cup_skill_centres_the_ball_before_closing():
    outcome, _hw, _ = run_skill(wrist_balls=((388, 292), (428, 292)))
    assert outcome.ok, outcome.error
    assert outcome.result["wrist_corrections_mm"][0][0] == pytest.approx(-16)


def test_ball_in_cup_releases_where_verified_releases_put_the_ball():
    segmentation = SceneSegmentation()
    segmentation.centres[("external", "mug")] = (382, 142)  # the mug pixel of a verified release
    outcome, hw, _ = run_skill(segmentation=segmentation)
    assert outcome.ok, outcome.error
    np.testing.assert_allclose(tool_xyz_mm(hw.release_pose)[:2], [236.5, 198.5], atol=8)


def test_ball_in_cup_fails_when_the_ball_is_seen_outside_the_mug():
    segmentation = SceneSegmentation()
    segmentation.centres[("external", "mug")] = (200, 100)
    segmentation.after_release = (385, 290)
    outcome, _hw, _ = run_skill(segmentation=segmentation)
    assert not outcome.ok and "outside the mug" in outcome.error.message


def test_ball_in_cup_skill_refuses_to_close_when_the_ball_stays_off_centre():
    outcome, hw, _ = run_skill(wrist_balls=((300, 292),))
    assert not outcome.ok and "not closing" in outcome.error.message
    assert hw.values["goals"]["gripper"] == pytest.approx(55)


def test_mcp_agent_runs_the_reviewed_ball_in_cup_program_in_one_call():
    pytest.importorskip("mcp")
    import json

    from cap_harness.so101.system2 import So101McpBridge, create_so101_app

    hw = SceneHardware()
    hw.values["positions"].update(dict(zip(JOINTS, [-5.0, -106.3, 66.7, 97.2, -84.8], strict=True)), gripper=2.0)
    segmentation = SceneSegmentation()
    segmentation.hardware = hw
    adapter = So101Adapter(hw, allow_motion=True, sleep=lambda _seconds: None,
                           segmentation_provider=segmentation, speed_deg_s=60.0)
    _app, session = create_so101_app(adapter)
    try:
        bridge = So101McpBridge(session)
        reply = bridge.call("cap_program_ball_in_cup", {})
        assert not reply.isError, reply.content[0].text
        assert json.loads(reply.content[0].text)["ok"] is True
        np.testing.assert_allclose(arm_goals(hw), [-5.0, -106.3, 66.7, 97.2, -84.8], atol=1e-6)
    finally:
        session.close()
