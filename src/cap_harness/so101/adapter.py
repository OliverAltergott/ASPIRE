"""Bounded SO101 control; no inferred camera geometry or automatic calibration.

The CAP boundary uses five arm angles in radians and a separate 0..1 gripper.
LeRobot uses degrees and a 0..100 gripper. Connecting is read-only. Explicit
enable primes measured positions before enabling torque, preserving servo gains.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import threading
import time

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    ExecutionResult,
    Observation,
    RobotState,
    TaskContext,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry

from .calibration import pixel_to_plane
from .kinematics import pitch_deg, plan_joint_path, plan_tool_path, solve_tool_pose, tool_xyz_mm

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MOTORS = (*JOINTS, "gripper")
PUBLIC_EXTENSIONS = frozenset({
    "so101.enable", "so101.move_delta", "so101.get_telemetry",
    "so101.move_tool", "so101.go_to", "so101.close_until_contact", "so101.detect_color",
    "so101.segment_text", "so101.pixel_to_base",
})
# Seeds for absolute tool targets: reach-forward configurations at shallow and steep pitch.
IK_SEEDS_DEG = ((0, -73, 85, 26), (0, -20, 75, -17), (0, 0, 40, 0), (0, -40, 20, 70))
# Loaded shoulder/elbow joints settle up to ~1.7 degrees below their goals under gravity.
SETTLE_TOLERANCE_DEG = 2.5
TICK_HZ = 50  # A full bus read takes ~11 ms; goals stream at this rate.
SETTLE_TICKS = 30  # At most 0.6 s waiting for the arm to arrive after a path.
MAX_TOOL_TRAVEL_MM = 200.0


class So101Adapter:
    """One owner of a calibrated bus and RGB cameras, with explicit motion gating.

    Joint/gripper primitives are bounded per call. so101.move_tool chains them along
    a straight line of the URDF-model tool point; observation pose fields stay
    empty because that model is not an independently verified calibration.
    """

    embodiment = "so101"
    arms = ("primary",)

    def __init__(self, hardware, *, allow_motion=False, sleep=time.sleep, event_sink=None,
                 table_floor_mm=-8.0, segmentation_provider=None, speed_deg_s=20.0, gripper_speed=50.0,
                 overhead_calibration=None):
        self.hardware = hardware
        self.overhead_calibration = overhead_calibration
        self.allow_motion = allow_motion
        self.speed_deg_s = speed_deg_s
        self.gripper_speed = gripper_speed
        self.segmentation_provider = segmentation_provider
        self.table_floor_mm = table_floor_mm
        self._sleep = sleep
        self._event_sink = event_sink or (lambda _event: None)
        self._lock = threading.RLock()
        self._goals = None
        self._last_observation = None
        self._closed = False
        self._commands = 0

    def _record(self, **event):
        self._event_sink({"timestamp_s": time.time(), **event})

    def get_task_context(self):
        return TaskContext(
            suite="operator", task_id=0, task_name="operator_goal", family=self.embodiment,
            language="Follow the operator goal and verify the physical outcome in both RGB views.",
            metadata={"arm_units": "radians", "gripper_units": "0 closed, 1 open",
                      "metric_camera_geometry": False, "end_effector_pose_available": False,
                      "max_joint_move_deg": 5, "max_gripper_step": 0.1,
                      "tool_frame": "URDF gripper_frame (jaw-tip centre) in so101_base mm",
                      "pitch": "shoulder_lift + elbow_flex + wrist_flex; 90 = jaws pointing down",
                      "table_floor_mm": self.table_floor_mm,
                      "exit_behavior": "preserve torque; no automatic home or release"},
        )

    def get_telemetry(self):
        with self._lock:
            result = self.hardware.read()
            measured = [result["positions"][j] for j in JOINTS]
            model = {"tool_xyz_mm": tool_xyz_mm(measured).round(1).tolist(), "pitch_deg": round(pitch_deg(measured), 1)}
            if self._goals is not None:
                model["goal_tool_xyz_mm"] = tool_xyz_mm([self._goals[j] for j in JOINTS]).round(1).tolist()
            return {**result, "urdf_model": model, "commands_sent": self._commands, "allow_motion": self.allow_motion}

    def _state(self, values):
        return RobotState(
            joint_positions=np.deg2rad([values["positions"][j] for j in JOINTS]),
            joint_velocities=np.zeros(5),  # Velocity is not measured by this adapter.
            end_effector_poses={},
            gripper_positions=values["positions"]["gripper"] / 100,
            joint_names=JOINTS, base_frame="so101_base", embodiment=self.embodiment,
            timestamp_s=values["timestamp_s"],
        )

    def get_robot_state(self):
        with self._lock:
            return self._state(self.hardware.read())

    def get_observation(self, camera_names=None):
        with self._lock:
            frames = self.hardware.frames()
            names = tuple(frames) if camera_names is None else (
                (camera_names,) if isinstance(camera_names, str) else tuple(camera_names)
            )
            if not names or any(name not in frames for name in names):
                raise ValueError("Unknown or empty camera selection")
            observation = Observation(
                cameras={name: CameraObservation(
                    rgb=frames[name][0], depth_m=None, intrinsics=None, camera_pose=None,
                    frame=f"{name}_optical", timestamp_s=frames[name][1],
                ) for name in names},
                robot_state=self.get_robot_state(), task_context=self.get_task_context(),
                timestamp_s=time.time(),
            )
            if set(names) == set(frames):
                self._last_observation = observation
            return observation

    def _check(self, *, enabled=True):
        if not self.allow_motion:
            raise PermissionError("SO101 physical motion is disabled; use --allow-motion")
        if self._last_observation is None or time.time() - self._last_observation.timestamp_s > 60:
            raise ValueError("Observe and inspect both RGB cameras before motion (maximum age 60 s)")
        values = self.hardware.read()
        if not values["calibrated"] or any(values["mode"][j] != 0 for j in MOTORS):
            raise ValueError("Existing calibration and position mode must match; no automatic writes")
        if not all(math.isfinite(values["positions"][j]) for j in MOTORS):
            raise ValueError("Nonfinite motor position")
        if enabled:
            if self._goals is None or any(values["torque"][j] != 1 for j in MOTORS):
                raise ValueError("Call so101.enable after inspecting the scene")
            if any(abs(values["goals"][j] - self._goals[j]) > 0.2 for j in MOTORS):
                raise ValueError("Motor goals changed outside this runtime")
            if any(abs(values["positions"][j] - self._goals[j]) > 5 for j in JOINTS):
                raise ValueError("Arm tracking error exceeds 5 degrees; inspect possible contact")
        # Check camera acquisition before writing even when the inspected image is still young.
        self.hardware.frames()
        return values

    @staticmethod
    def _failure(exc, steps=0, observation=None):
        code = ErrorCode.SAFETY_INTERLOCK if isinstance(exc, PermissionError) else ErrorCode.EXECUTION_FAILED
        return ExecutionResult(ok=False, steps_executed=steps, final_observation=observation,
                               error=ApiError(code=code, message=str(exc)))

    def enable(self):
        """Prime a hold at measured positions before enabling torque; never home."""
        with self._lock:
            try:
                values = self._check(enabled=False)
                torque = set(values["torque"].values())
                if torque == {0}:
                    goals = dict(values["positions"])
                    self._record(operation="prime_hold", goals_degrees_percent=goals)
                    self.hardware.write(goals)
                    self.hardware.enable()
                elif torque == {1}:
                    goals = dict(values["goals"])
                    if any(abs(goals[j] - values["positions"][j]) > 5 for j in JOINTS):
                        raise ValueError("Powered arm has a large tracking error; refusing to rebase goals")
                else:
                    raise ValueError("Mixed torque state; refusing partial-arm activation")
                self._goals = goals
                self._check()
                return ExecutionResult(ok=True, steps_executed=0, final_observation=self.get_observation())
            except Exception as exc:
                return self._failure(exc)

    def _gripper_path(self, target):
        """Gripper waypoints from the current goal to target, at most 10 units apart."""
        count = max(1, math.ceil(abs(target - self._goals["gripper"]) / 10))
        return [{"gripper": self._goals["gripper"] + (target - self._goals["gripper"]) * i / count}
                for i in range(1, count + 1)]

    def _move(self, path, *, gripper_speed=None):
        """Stream goals through waypoints (dicts of motor targets) in one continuous motion.

        Every waypoint is checked before the first write: calibrated limits, at most 5 degrees
        (gripper 10 units) from the previous waypoint, and the first within 5 degrees of the
        measured arm. Each tick re-checks torque and arm tracking; the arm settles once.
        """
        path = [path] if isinstance(path, Mapping) else list(path)
        steps = 0
        try:
            before = self._check()
            if not path or any(not changes or any(j not in MOTORS for j in changes) for changes in path):
                raise ValueError("Expected named SO101 joint targets")
            previous = dict(self._goals)
            for changes in path:
                for joint, target in changes.items():
                    lower, upper = before["limits"][joint]
                    bound = 10 if joint == "gripper" else 5
                    if not math.isfinite(target) or not lower <= target <= upper:
                        raise ValueError(f"{joint}: target outside calibrated limits")
                    if abs(target - previous[joint]) > bound + 1e-8:
                        raise ValueError(f"{joint}: requested move exceeds {bound} units")
                previous.update(changes)
            for joint, target in path[0].items():
                if joint != "gripper" and abs(target - before["positions"][joint]) > 5:
                    raise ValueError(f"{joint}: target exceeds measured-position tracking bound")
            # Servo lag grows with speed; settling below still demands SETTLE_TOLERANCE_DEG.
            track_bound = 5 + 0.1 * self.speed_deg_s
            # Consume evidence even if a subsequent command or capture fails.
            self._last_observation = None
            peak = 0.0
            for changes in path:
                start = {j: self._goals[j] for j in changes}
                duration = max(abs(changes[j] - start[j]) / (
                    (gripper_speed or self.gripper_speed) if j == "gripper" else self.speed_deg_s) for j in changes)
                count = max(1, math.ceil(duration * TICK_HZ))
                for i in range(1, count + 1):
                    tick = time.perf_counter()
                    current = self.hardware.read()
                    if any(current["torque"][j] != 1 for j in MOTORS):
                        raise ValueError("Torque was disabled during motion")
                    lag = max(abs(current["positions"][j] - self._goals[j]) for j in JOINTS)
                    if lag > track_bound:
                        raise ValueError("Tracking error during motion; stopped future commands")
                    peak = max(peak, lag)
                    goals = {j: start[j] + (changes[j] - start[j]) * i / count for j in changes}
                    self._record(operation="command", goals_degrees_percent=goals)
                    self.hardware.write(goals)
                    self._goals.update(goals)
                    self._commands += 1
                    steps += 1
                    self._sleep(max(0.0, duration / count - (time.perf_counter() - tick)))
            targets = {joint: value for changes in path for joint, value in changes.items()}
            for _ in range(SETTLE_TICKS):
                after = self.hardware.read()
                errors = {j: abs(after["positions"][j] - targets[j]) for j in targets}
                if all(error <= (2 if j == "gripper" else 0.5 * SETTLE_TOLERANCE_DEG) for j, error in errors.items()):
                    break
                self._sleep(1 / TICK_HZ)
            observation = self.get_observation()
            after = self.hardware.read()
            errors = {j: abs(after["positions"][j] - targets[j]) for j in targets}
            ok = all(error <= (2 if j == "gripper" else SETTLE_TOLERANCE_DEG) for j, error in errors.items())
            self._record(operation="terminal", telemetry=after, target_errors=errors)
            return ExecutionResult(
                ok=ok, steps_executed=steps, final_observation=observation,
                error=None if ok else ApiError(ErrorCode.CONTROLLER_FAILED,
                    "Target not reached; inspect contact before continuing. Gripper stall alone is not a grasp."),
                diagnostics={"target_errors_degrees_percent": errors, "commands_sent": self._commands,
                             "peak_tracking_error_deg": round(peak, 2)},
            )
        except Exception as exc:
            self._record(operation="failure", message=str(exc), steps_executed=steps)
            return self._failure(exc, steps)

    def move_delta(self, deltas_deg: Mapping[str, float]):
        """Move named arm joints by at most 5 degrees, preserving untouched goals."""
        with self._lock:
            if self._goals is None:
                return self._failure(ValueError("Enable the arm first"))
            if not deltas_deg or any(j not in JOINTS for j in deltas_deg):
                return self._failure(ValueError("Use arm joint names; set_gripper controls the gripper"))
            return self._move({j: self._goals[j] + float(d) for j, d in deltas_deg.items()})

    def move_to_joints(self, target: list[float], *, arm: str = "primary"):
        """Absolute five-joint targets in radians; each move is bounded to 5 degrees."""
        with self._lock:
            values = np.asarray(target, dtype=float)
            if arm != "primary" or values.shape != (5,) or not np.isfinite(values).all():
                return self._failure(ValueError("Expected primary arm and five finite radians"))
            return self._move(dict(zip(JOINTS, np.rad2deg(values), strict=True)))

    def set_gripper(self, position: float, *, arm: str = "primary"):
        """Set opening from 0 closed to 1 open in checked steps of at most 0.1."""
        with self._lock:
            if arm != "primary" or not math.isfinite(position) or not 0 <= position <= 1:
                return self._failure(ValueError("Expected primary arm and gripper in [0, 1]"))
            if self._goals is None:
                return self._move({"gripper": position * 100})  # Refused by the enable interlock.
            return self._move(self._gripper_path(position * 100))

    def move_tool(self, delta_mm, pitch=None):
        """Move the jaw-tip tool point along a straight so101_base line by (dx, dy, dz) mm.

        pitch optionally changes the jaw angle below horizontal (90 = pointing down) along
        the way. Planned from commanded goals as bounded joint steps that stop at the first
        unsettled step; no planned tool point goes below the table floor. This checks the
        tool point only, not collisions of the arm, fingers or held object.
        """
        with self._lock:
            try:
                if self._goals is None:
                    raise ValueError("Enable the arm first")
                delta = np.asarray(delta_mm, dtype=float)
                if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > MAX_TOOL_TRAVEL_MM:
                    raise ValueError(f"Expected three finite millimetres, at most {MAX_TOOL_TRAVEL_MM:g} mm in total")
                if pitch is not None and not (math.isfinite(pitch) and 0 <= pitch <= 90):
                    raise ValueError("pitch must lie in [0, 90] degrees")
                limits = self.hardware.read()["limits"]
                start = [self._goals[j] for j in JOINTS]
                waypoints = plan_tool_path(start, delta, pitch, [limits[j] for j in JOINTS],
                                           floor_z_mm=self.table_floor_mm)
            except Exception as exc:
                return self._failure(exc)
            result = self._move([dict(zip(JOINTS, waypoint, strict=True)) for waypoint in waypoints])
            steps = result.steps_executed
            diagnostics = {**result.diagnostics, "waypoints": len(waypoints),
                           "target_tool_xyz_mm": (tool_xyz_mm(start) + delta).round(1).tolist()}
            if result.final_observation is not None:
                measured = np.rad2deg(result.final_observation.robot_state.joint_positions["primary"])
                diagnostics["tool_xyz_mm"] = tool_xyz_mm(measured).round(1).tolist()
            return ExecutionResult(ok=result.ok, steps_executed=steps, final_observation=result.final_observation,
                                   error=result.error, diagnostics=diagnostics)

    def go_to(self, xyz_mm=None, pitch=None, joints_deg=None):
        """Joint-space move to an absolute tool target (xyz_mm and pitch) or five arm joints.

        Use it to leave or return to folded poses such as rest, where straight tool lines
        are infeasible. Steps are <=3 degrees and the tool point stays above the table floor
        along the whole path. Wrist roll is kept for tool targets. Not collision checking.
        """
        with self._lock:
            try:
                if self._goals is None:
                    raise ValueError("Enable the arm first")
                limits = self.hardware.read()["limits"]
                start = np.array([self._goals[j] for j in JOINTS])
                if joints_deg is not None:
                    target = np.asarray(joints_deg, dtype=float)
                    if target.shape != (5,) or not np.isfinite(target).all():
                        raise ValueError("joints_deg needs five finite arm angles in degrees")
                else:
                    xyz = np.asarray(xyz_mm, dtype=float)
                    if xyz.shape != (3,) or not np.isfinite(xyz).all() or pitch is None or not 0 <= pitch <= 90:
                        raise ValueError("Give xyz_mm (three finite mm) with pitch in [0, 90], or joints_deg")
                    seeds = [start] + [(start[0], *seed[1:], start[4]) for seed in IK_SEEDS_DEG]
                    target = solve_tool_pose(xyz, pitch, [limits[j] for j in JOINTS], seeds)
                for joint, angle in zip(JOINTS, target, strict=True):
                    lower, upper = limits[joint]
                    if not lower <= angle <= upper:
                        raise ValueError(f"{joint}: {angle:.1f} deg is outside calibrated limits")
                waypoints = plan_joint_path(start, target, floor_z_mm=self.table_floor_mm)
            except Exception as exc:
                return self._failure(exc)
            result = self._move([dict(zip(JOINTS, waypoint, strict=True)) for waypoint in waypoints])
            steps = result.steps_executed
            return ExecutionResult(ok=result.ok, steps_executed=steps, final_observation=result.final_observation,
                                   error=result.error, diagnostics={**result.diagnostics, "waypoints": len(waypoints),
                                   "target_joints_deg": np.round(target, 1).tolist(),
                                   "target_tool_xyz_mm": tool_xyz_mm(target).round(1).tolist()})

    def close_until_contact(self, min_opening: float = 0.1, ease: float = 0.025, speed: float = 20.0):
        """Close at `speed` (units per second) until the jaws stall on an object or reach min_opening.

        A fast sweep can knock a light ball out from between the jaws; 20 units/s matched
        the stepwise closes that grasped it.

        A stall more than 2 units above the commanded step is contact; the goal is then
        eased to `ease` below the stalled opening to limit squeeze. ok=False with
        contact=False means the jaws closed without touching anything. Contact alone is
        not a grasp: lift and confirm the object moves with the gripper.
        """
        with self._lock:
            if self._goals is None:
                return self._failure(ValueError("Enable the arm first"))
            if not (math.isfinite(min_opening) and 0 <= min_opening <= 1 and math.isfinite(ease) and 0 <= ease <= 0.1):
                return self._failure(ValueError("Expected min_opening in [0, 1] and ease in [0, 0.1]"))
            target = min_opening * 100
            if target >= self._goals["gripper"]:
                return self._failure(ValueError("Gripper is already closed beyond min_opening"))
            result = self._move(self._gripper_path(target), gripper_speed=speed)
            if result.ok:
                return ExecutionResult(
                    ok=False, steps_executed=result.steps_executed, final_observation=result.final_observation,
                    error=ApiError(ErrorCode.CONTROLLER_FAILED, f"Closed to {min_opening:.2f} without contact"),
                    diagnostics={"contact": False, "opening": self.hardware.read()["positions"]["gripper"] / 100},
                )
            opening = self.hardware.read()["positions"]["gripper"]
            if result.error.code != ErrorCode.CONTROLLER_FAILED or opening <= target + 2:
                return result
            hold = max(target, opening - ease * 100)
            final = self._move(self._gripper_path(hold)) if hold > target else result
            return ExecutionResult(
                ok=True, steps_executed=result.steps_executed + final.steps_executed,
                final_observation=final.final_observation,
                diagnostics={"contact": True, "opening": opening / 100, "held_goal": hold / 100,
                             "gripper_load": self.hardware.read()["loads"]["gripper"],
                             "note": "Contact alone is not a grasp; lift and verify the object follows."},
            )

    def detect_color(self, hsv_lower, hsv_upper, camera_names=None, min_radius_px: float = 4.0,
                     max_radius_px: float = 1e9, min_fill: float = 0.0):
        """Find colour blobs in a fresh observation, using OpenCV HSV (H 0-179, S and V 0-255).

        Returns {camera: [{"center_px", "radius_px", "area_px", "fill"}]}, largest first and at
        most five per camera, so objects can be tracked numerically between image checks.
        fill is blob area over its enclosing circle; min_fill ~0.6 keeps round objects.
        """
        import cv2

        lower, upper = np.asarray(hsv_lower), np.asarray(hsv_upper)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("hsv_lower and hsv_upper need three values each")
        observation = self.get_observation()
        names = observation.cameras if camera_names is None else (
            (camera_names,) if isinstance(camera_names, str) else tuple(camera_names))
        blobs = {}
        for name in names:
            hsv = cv2.cvtColor(np.ascontiguousarray(observation.cameras[name].rgb), cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            found = []
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                area = float(cv2.contourArea(contour))
                if min_radius_px <= radius <= max_radius_px and area >= min_fill * math.pi * radius ** 2:
                    found.append({"center_px": [round(x, 1), round(y, 1)], "radius_px": round(radius, 1),
                                  "area_px": area, "fill": round(area / (math.pi * radius ** 2), 2)})
            blobs[name] = sorted(found, key=lambda blob: blob["area_px"], reverse=True)[:5]
        return blobs

    def segment_text(self, query, camera_names=None, min_score: float = 0.3):
        """Text-prompted 2D detection (SAM3) in a fresh observation of each camera.

        Returns {camera: [{"score", "box_xyxy", "center_px", "area_px"}]}, best first and at
        most three per camera; center_px is the mask centroid. RGB only: no depth or 3D.
        A list of queries returns {query: {camera: [...]}} from one observation; the SAM3
        service encodes each image once, so extra queries cost under a second.
        """
        observation = self.get_observation()
        names = observation.cameras if camera_names is None else (
            (camera_names,) if isinstance(camera_names, str) else tuple(camera_names))
        queries = [query] if isinstance(query, str) else list(query)
        found = {q: {} for q in queries}
        for name in names:
            for q in queries:
                found[q][name] = self._detections(observation.cameras[name], q, min_score)
        return found[query] if isinstance(query, str) else found

    def _detections(self, camera, query, min_score):
        result = self.segmentation_provider.segment_text(camera, query)
        if not result.ok:
            if result.error.code != ErrorCode.NO_SEGMENTATION:
                raise RuntimeError(f"{camera.frame}: {result.error.message}")
            return []
        found = []
        for segmentation in result.segmentations:
            rows, cols = np.nonzero(segmentation.mask)
            if segmentation.score >= min_score and rows.size:
                found.append({"score": round(float(segmentation.score), 3),
                              "box_xyxy": np.round(segmentation.box_xyxy, 1).tolist(),
                              "center_px": [round(float(cols.mean()), 1), round(float(rows.mean()), 1)],
                              "area_px": int(rows.size)})
        return found[:3]

    def pixel_to_base(self, pixel, plane: str = "table"):
        """External-camera pixel to so101_base [x, y] mm on a calibrated plane.

        "table" is a ball resting on the table; "rim" is a ball held at mug-rim height.
        """
        calibration = self.overhead_calibration["planes"][plane]
        return [round(value, 1) for value in pixel_to_plane(calibration["homography"], pixel)]

    def register_tools(self):
        registry = ToolRegistry(public_extension_allowlist=PUBLIC_EXTENSIONS)
        for name, function in {
            "get_task_context": self.get_task_context, "get_robot_state": self.get_robot_state,
            "get_observation": self.get_observation, "move_to_joints": self.move_to_joints,
            "set_gripper": self.set_gripper, "so101.enable": self.enable,
            "so101.move_delta": self.move_delta, "so101.get_telemetry": self.get_telemetry,
            "so101.move_tool": self.move_tool, "so101.go_to": self.go_to,
            "so101.close_until_contact": self.close_until_contact,
            "so101.detect_color": self.detect_color,
            **({"so101.segment_text": self.segment_text} if self.segmentation_provider is not None else {}),
            **({"so101.pixel_to_base": self.pixel_to_base} if self.overhead_calibration is not None else {}),
        }.items():
            registry.register(function, name=name, layer="so101", capability="joint-space-rgb", public=True)
        return registry

    def close(self):
        with self._lock:
            if not self._closed:
                self.hardware.close()  # Preserve torque: disconnect must not drop a held object.
                self._closed = True
