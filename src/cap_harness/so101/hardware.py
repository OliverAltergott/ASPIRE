"""LeRobot 0.4.4 transport; imports stay lazy for hardware-free tests."""

from __future__ import annotations

import fcntl
from pathlib import Path
import time

from .adapter import MOTORS


class LeRobotHardware:
    def __init__(self, *, port, robot_id, calibration_dir, cameras):
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower, SOFollowerRobotConfig

        if len(cameras) != 2 or len(set(cameras.values())) != 2:
            raise ValueError("Two distinct RGB cameras are required")
        directory = Path.home() / ".cache/so101-direct"
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = (directory / "hardware.lock").open("a")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.robot = SO101Follower(SOFollowerRobotConfig(
            port=port, id=robot_id, calibration_dir=Path(calibration_dir), cameras={}, use_degrees=True,
        ))
        self.cameras = {}
        self.camera_reconnects = {}
        try:
            if set(self.robot.calibration) != set(MOTORS):
                raise ValueError("A complete existing SO101 calibration is required")
            self.robot.bus.connect()  # robot.connect() configures servos and enables torque.
            if not self.robot.is_calibrated:
                raise ValueError("Motor calibration does not match the saved calibration")
            for name, index in cameras.items():
                camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30))
                self.cameras[name] = camera
                camera.connect()
        except BaseException:
            self.close()
            raise

    def read(self):
        bus = self.robot.bus
        result = {key: bus.sync_read(register, normalize=normalize, num_retry=2)
                  for key, register, normalize in (
                      ("positions", "Present_Position", True), ("goals", "Goal_Position", True),
                      ("loads", "Present_Load", False), ("torque", "Torque_Enable", False),
                      ("mode", "Operating_Mode", False),
                  )}
        result["calibrated"] = bus.is_calibrated
        result["camera_reconnects"] = dict(self.camera_reconnects)
        result["limits"] = {}
        for joint, calibration in self.robot.calibration.items():
            result["limits"][joint] = sorted(bus._normalize({calibration.id: raw})[calibration.id]
                                              for raw in (calibration.range_min, calibration.range_max))
        result["timestamp_s"] = time.time()
        return result

    def frames(self):
        # LeRobot ends a camera's read thread after 11 consecutive read errors (USB glitch,
        # moved camera). Restart it here; motion still needs fresh frames from every camera.
        # Check the views after a reconnect: a replugged camera can come back on another index.
        for name, camera in self.cameras.items():
            if camera.thread is None or not camera.thread.is_alive():
                camera.disconnect()
                camera.connect()
                self.camera_reconnects[name] = self.camera_reconnects.get(name, 0) + 1
        # Require frames acquired after this request, including after any preceding motion.
        started = time.perf_counter()
        result = {}
        while len(result) != len(self.cameras):
            for name, camera in self.cameras.items():
                with camera.frame_lock:
                    frame, timestamp = camera.latest_frame, camera.latest_timestamp
                    if frame is not None and timestamp is not None and timestamp > started:
                        result[name] = (frame.copy(), time.time() - (time.perf_counter() - timestamp))
            if time.perf_counter() - started > 2:
                raise TimeoutError("Missing fresh camera frames; no motion permitted")
            if len(result) != len(self.cameras):
                time.sleep(0.005)
        if max(t for _, t in result.values()) - min(t for _, t in result.values()) > 0.2:
            raise TimeoutError("Camera host timestamps differ by more than 200 ms")
        return result

    def latest_frames(self):
        """Newest frame per camera without waiting; for recording only, never for motion checks."""
        result = {}
        for name, camera in self.cameras.items():
            with camera.frame_lock:
                if camera.latest_frame is not None:
                    result[name] = camera.latest_frame.copy()
        return result

    def write(self, goals):
        self.robot.bus.sync_write("Goal_Position", goals)

    def enable(self):
        self.robot.bus.enable_torque()

    def close(self):
        try:
            for camera in self.cameras.values():
                if camera.is_connected or camera.thread is not None:
                    camera.disconnect()
        finally:
            try:
                if self.robot.bus.is_connected:
                    self.robot.bus.disconnect(disable_torque=False)
            finally:
                self._lock.close()
