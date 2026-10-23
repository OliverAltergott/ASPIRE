"""Run registered SO101 tools through ASPIRE's ProgramExecutor.

Use --program FILE for a single program or --stdio for newline-delimited JSON
requests {"program": "result = get_observation()"}. Each response names the saved
CAP trace, state, fresh camera images and view.jpg (all cameras side by side, for
one image read per check). Startup never enables torque. Agent shells that cannot
hold a pipe open can follow a file: ``tail -n +1 -f commands.jsonl | python -m
cap_harness.so101 ... --stdio > replies.jsonl``. --mcp serves the same tools to a
coding agent through ASPIRE's System 2 MCP session (cap_harness.so101.system2), and
--sam3-url adds so101.segment_text. --record writes a continuous video.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from cap_harness.runtime import ProgramExecutor

from .adapter import So101Adapter
from .hardware import LeRobotHardware


def summary(value):
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim <= 1 else {"shape": list(value.shape)}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: summary(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summary(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Recorder(threading.Thread):
    """Appends the newest frames of all cameras side by side at a fixed rate (H.264)."""

    def __init__(self, hardware, path, fps=10):
        import imageio.v2 as imageio

        super().__init__(daemon=True)
        self.hardware, self.period, self.stopped = hardware, 1 / fps, threading.Event()
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p")

    def run(self):
        while not self.stopped.wait(self.period):
            frames = self.hardware.latest_frames()
            if len(frames) == len(self.hardware.cameras):
                self.writer.append_data(np.hstack([frames[name] for name in sorted(frames)]))

    def close(self):
        self.stopped.set()
        self.join()
        self.writer.close()


def run(adapter, source, output):
    from PIL import Image

    directory = output / str(time.time_ns())
    directory.mkdir(parents=True)
    (directory / "program.py").write_text(source)
    result = ProgramExecutor(adapter.register_tools()).execute_program(source)
    payload = summary(result)
    payload["artifact_dir"] = str(directory)
    observation = getattr(result.result, "final_observation", None)
    if observation is None and hasattr(result.result, "cameras"):
        observation = result.result
    if observation is not None:
        payload["images"], panels = {}, []
        for name, camera in observation.cameras.items():
            path = directory / f"{name}.png"
            image = Image.fromarray(camera.rgb)
            image.save(path)
            payload["images"][name] = str(path)
            panels.append(image.resize((image.width * 360 // image.height, 360)))
        view = Image.new("RGB", (sum(panel.width for panel in panels), 360))
        for index, panel in enumerate(panels):
            view.paste(panel, (sum(p.width for p in panels[:index]), 0))
        view.save(directory / "view.jpg", quality=80)
        payload["view"] = str(directory / "view.jpg")
    (directory / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, allow_nan=False), flush=True)
    return result.ok and bool(getattr(result.result, "ok", True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--cameras", type=json.loads, required=True, help='JSON object, e.g. {"external":1,"wrist":0}')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--record", action="store_true", help="write OUTPUT/session-<time>.mp4 at 10 fps")
    parser.add_argument("--sam3-url", help="SAM3 service, e.g. http://127.0.0.1:8114")
    parser.add_argument("--mcp-port", type=int, default=8222)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--program", type=Path)
    mode.add_argument("--stdio", action="store_true")
    mode.add_argument("--mcp", action="store_true", help="serve MCP at http://127.0.0.1:PORT/mcp/")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    hardware = LeRobotHardware(port=args.port, robot_id=args.robot_id,
                              calibration_dir=args.calibration_dir.expanduser(), cameras=args.cameras)
    recorder = None
    with (output / "commands.jsonl").open("a", buffering=1) as log:
        segmentation = None
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            segmentation = Sam3Provider(base_url=args.sam3_url)
        adapter = So101Adapter(hardware, allow_motion=args.allow_motion, segmentation_provider=segmentation,
                              event_sink=lambda e: log.write(json.dumps(e, allow_nan=False) + "\n"))
        try:
            if args.record:
                recorder = Recorder(hardware, output / f"session-{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                recorder.start()
            if args.program:
                return 0 if run(adapter, args.program.read_text(), output) else 1
            if args.mcp:
                import uvicorn

                from .system2 import create_so101_app

                with (output / "tools.jsonl").open("a", buffering=1) as tools:
                    app, session = create_so101_app(
                        adapter, tool_result_sink=lambda **r: tools.write(json.dumps(r, default=str) + "\n"))
                    try:
                        uvicorn.run(app, host="127.0.0.1", port=args.mcp_port)
                    finally:
                        session.close()
                return 0
            print(json.dumps({"ready": True, "motion_authorized": args.allow_motion,
                              "torque": "unchanged", "tools": list(adapter.register_tools().public_tools())}), flush=True)
            for line in sys.stdin:
                try:
                    run(adapter, json.loads(line)["program"], output)
                except Exception as exc:
                    print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        except KeyboardInterrupt:
            pass  # SIGINT is the normal way to stop a session; torque is preserved.
        finally:
            if recorder:
                recorder.close()
            adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
