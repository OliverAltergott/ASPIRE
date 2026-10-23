"""Overhead-camera calibration from a held ball: one table-plane homography per height.

Records pair the ball's external-camera pixel with the joints that held it. The ball
centre is the URDF tool point moved 18 mm back along the jaw axis, where the
ball-into-cup grasp holds it. Each plane ("table", "rim") gets a homography from
pixels to so101_base millimetres at that plane's mean height.

    python -m cap_harness.so101.calibration RESULT_JSON OUT_JSON

RESULT_JSON is a program result.json whose result has "calibration_records": a list of
{"plane", "joints_deg", "pixel"}. Pass OUT_JSON to the runtime as --overhead-calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from .kinematics import tool_transform

BALL_BEHIND_TIP_MM = 18.0


def ball_centre_mm(joints_deg):
    transform = tool_transform(joints_deg)
    return transform[:3, 3] * 1000 - BALL_BEHIND_TIP_MM * transform[:3, 2]


def fit_homography(pixels, points):
    """Least-squares pixel -> plane homography (normalised DLT); needs four or more pairs."""
    pixels, points = np.asarray(pixels, float), np.asarray(points, float)
    if len(pixels) < 4 or pixels.shape != points.shape or pixels.shape[1] != 2:
        raise ValueError("Need at least four pixel/point pairs")

    def normaliser(values):
        centre, scale = values.mean(0), np.sqrt(2) / np.mean(np.linalg.norm(values - values.mean(0), axis=1))
        return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])

    to_pixel, to_point = normaliser(pixels), normaliser(points)
    rows = []
    for (u, v, _), (x, y, _) in zip(np.c_[pixels, np.ones(len(pixels))] @ to_pixel.T,
                                     np.c_[points, np.ones(len(points))] @ to_point.T, strict=True):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    homography = np.linalg.inv(to_point) @ np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3) @ to_pixel
    return homography / homography[2, 2]


def pixel_to_plane(homography, pixel):
    x, y, w = np.asarray(homography, float) @ [pixel[0], pixel[1], 1.0]
    return [x / w, y / w]


def fit_records(records):
    planes = {}
    for name in sorted({record["plane"] for record in records}):
        chosen = [record for record in records if record["plane"] == name]
        centres = np.array([ball_centre_mm(record["joints_deg"]) for record in chosen])
        pixels = np.array([record["pixel"] for record in chosen], float)
        homography = fit_homography(pixels, centres[:, :2])
        residuals = [np.linalg.norm(np.subtract(pixel_to_plane(homography, p), c[:2])) for p, c in zip(pixels, centres)]
        planes[name] = {"z_mm": round(float(centres[:, 2].mean()), 1), "homography": homography.tolist(),
                        "points": len(chosen), "rms_mm": round(float(np.sqrt(np.mean(np.square(residuals)))), 2)}
    return {"camera": "external", "ball_behind_tip_mm": BALL_BEHIND_TIP_MM, "planes": planes}


def main():
    result = json.loads(Path(sys.argv[1]).read_text())
    records = (result.get("result") or result)["calibration_records"]
    calibration = fit_records(records)
    Path(sys.argv[2]).write_text(json.dumps(calibration, indent=2) + "\n")
    for name, plane in calibration["planes"].items():
        print(f"{name}: {plane['points']} points at z={plane['z_mm']} mm, rms {plane['rms_mm']} mm")


if __name__ == "__main__":
    main()
