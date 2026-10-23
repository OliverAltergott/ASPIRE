"""SO101 forward kinematics and straight-line tool planning from the URDF model.

Joint origins follow TheRobotStudio/SO-ARM100 ``so101_new_calib.urdf``; angles are
LeRobot calibrated degrees. The tool point is the URDF ``gripper_frame``, near the
centre of the jaw tips. On the robot this was validated only loosely: in 26 of 31
recorded ball grasps the tool point sat at z = -10..+5 mm. Treat positions as a
model, not a measured calibration.
"""

from __future__ import annotations

import numpy as np

TOOL_PATH_STEP_DEG = 3.0  # Leaves margin under the adapter's 5 degree goal/measured bounds.


def _transform(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    result = np.eye(4)
    result[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]]
    result[:3, 3] = xyz
    return result


_JOINT_ORIGINS = [_transform(xyz, rpy) for xyz, rpy in (
    ((0.0388, 0.0, 0.0624), (np.pi, 0.0, -np.pi)),           # shoulder_pan
    ((-0.0304, -0.0183, -0.0542), (-np.pi / 2, -np.pi / 2, 0)),  # shoulder_lift
    ((-0.1126, -0.028, 0.0), (0.0, 0.0, np.pi / 2)),         # elbow_flex
    ((-0.1349, 0.0052, 0.0), (0.0, 0.0, -np.pi / 2)),        # wrist_flex
    ((0.0, -0.0611, 0.0181), (np.pi / 2, 0.0487, np.pi)),     # wrist_roll
)]
_TOOL = _transform((-0.0079, -0.0002, -0.0981), (0.0, np.pi, 0.0))


def tool_transform(arm_deg):
    """Base-to-tool transform in metres. Tool z points from the wrist out of the jaws;
    tool y is the jaw-plane normal (the wrist camera's side); tool x opens the jaws.
    """
    result = np.eye(4)
    for origin, angle in zip(_JOINT_ORIGINS, np.radians(np.asarray(arm_deg, float)[:5]), strict=True):
        rotation = np.eye(4)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        result = result @ origin @ rotation
    return result @ _TOOL


def tool_xyz_mm(arm_deg):
    return tool_transform(arm_deg)[:3, 3] * 1000


def pitch_deg(arm_deg):
    """Jaw angle below horizontal: 0 points them outward, 90 points them straight down."""
    return float(np.sum(np.asarray(arm_deg, float)[1:4]))


def _solve(seed, xyz_mm, pitch, limits):
    from scipy.optimize import least_squares

    def residual(arm):
        pose = np.concatenate((arm, seed[4:5]))
        return np.append(tool_xyz_mm(pose) - xyz_mm, 3 * (arm[1:4].sum() - pitch))

    solution = least_squares(residual, seed[:4], bounds=(limits[:4, 0], limits[:4, 1]), xtol=1e-10, ftol=1e-10)
    pose = np.concatenate((solution.x, seed[4:5]))
    if np.linalg.norm(tool_xyz_mm(pose) - xyz_mm) > 0.5 or abs(pitch_deg(pose) - pitch) > 0.2:
        raise ValueError(f"Tool target {np.round(xyz_mm, 1).tolist()} at pitch {pitch:.1f} is unreachable")
    return pose


def plan_tool_path(start_deg, delta_mm, pitch, limits_deg, *, floor_z_mm):
    """Joint waypoints (degrees, five arm joints) for a straight tool line plus pitch change.

    Wrist roll is held. Consecutive waypoints differ by at most TOOL_PATH_STEP_DEG per
    joint, and every densely sampled tool point stays at or above ``floor_z_mm``.
    """
    start = np.asarray(start_deg, float)[:5]
    limits = np.asarray(limits_deg, float)[:5]
    origin, start_pitch = tool_xyz_mm(start), pitch_deg(start)
    pitch = start_pitch if pitch is None else float(pitch)
    samples = max(20, int(np.ceil(max(np.linalg.norm(delta_mm), abs(pitch - start_pitch)))) * 2)
    dense, pose = [], start
    for t in np.linspace(0, 1, samples + 1)[1:]:
        pose = _solve(pose, origin + t * np.asarray(delta_mm, float), start_pitch + t * (pitch - start_pitch), limits)
        if tool_xyz_mm(pose)[2] < floor_z_mm:
            raise ValueError(f"Tool path goes below the table floor (z < {floor_z_mm} mm)")
        dense.append(pose)
    waypoints, last = [], start
    for previous, pose in zip([start, *dense[:-1]], dense, strict=True):
        if np.abs(pose - last).max() > TOOL_PATH_STEP_DEG:
            if np.abs(previous - last).max() > TOOL_PATH_STEP_DEG or previous is last:
                raise ValueError("Tool path needs a joint jump; split the move or change pitch separately")
            waypoints.append(previous)
            last = previous
    waypoints.append(dense[-1])
    return waypoints


def solve_tool_pose(xyz_mm, pitch, limits_deg, seeds_deg):
    """Arm joints (degrees) for a tool position and pitch, keeping each seed's wrist roll.

    Tries every seed and returns the solution closest to the first one.
    """
    limits = np.asarray(limits_deg, float)[:5]
    solutions = []
    for seed in seeds_deg:
        try:
            solutions.append(_solve(np.asarray(seed, float)[:5], np.asarray(xyz_mm, float), float(pitch), limits))
        except ValueError:
            continue
    if not solutions:
        raise ValueError(f"Tool target {np.round(xyz_mm, 1).tolist()} at pitch {pitch:.1f} is unreachable")
    first = np.asarray(seeds_deg[0], float)[:5]
    return min(solutions, key=lambda pose: np.abs(pose - first).max())


def plan_joint_path(start_deg, target_deg, *, floor_z_mm):
    """Linear joint-space waypoints, <=TOOL_PATH_STEP_DEG apart, tool point above the floor."""
    start, target = np.asarray(start_deg, float)[:5], np.asarray(target_deg, float)[:5]
    for t in np.linspace(0, 1, 101):
        if tool_xyz_mm(start + t * (target - start))[2] < floor_z_mm:
            raise ValueError(f"Joint path goes below the table floor (z < {floor_z_mm} mm)")
    count = max(1, int(np.ceil(np.abs(target - start).max() / TOOL_PATH_STEP_DEG)))
    return [start + (target - start) * i / count for i in range(1, count + 1)]
