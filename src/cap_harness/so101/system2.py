"""Serve the SO101 adapter through ASPIRE's System 2 session and MCP transport.

The session, job model, operator steering and MCP transport are the shared System 2
ones; this module supplies the SO101 tool allowlist and two SO101 choices:

- ``observe`` takes a fresh adapter observation, which is the inspection the motion
  interlock requires, and returns both cameras as one side-by-side image;
- motion results return text only, so images arrive when the agent asks for them.
  Hosted models cap images per request; per-step images ended an earlier session.

The reviewed ``ball_in_cup`` program is offered as ``cap_program_ball_in_cup``.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Any

import numpy as np

from cap_harness.agent.mcp import CapMcpBridge
from cap_harness.agent.program_catalog import (
    CapProgramCatalog,
    CapProgramDefinition,
    LoadedCapProgram,
)
from cap_harness.agent.session import LiveAgentSession

DIRECT_TOOLS = frozenset({
    "get_robot_state", "move_to_joints", "set_gripper",
    "so101.enable", "so101.get_telemetry", "so101.move_delta", "so101.move_tool", "so101.go_to",
    "so101.close_until_contact", "so101.detect_color", "so101.segment_text", "so101.pixel_to_base",
})
IMAGE_TOOLS = frozenset({"observe", "get_job"})


class So101SessionApi:
    """The part of CapApi that LiveAgentSession and CapMcpBridge read."""

    default_camera = None

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    @property
    def configured_providers(self) -> dict[str, object]:
        return {"segmentation": self.adapter.segmentation_provider is not None, "ik": (),
                "trajectory_planning": (), "pose_planning": ()}

    def get_robot_state(self) -> Any:
        return self.adapter.get_robot_state()


class So101Programs(CapProgramCatalog):
    """Reviewed SO101 programs; YAM's Home and observe programs do not apply here."""

    def __init__(self) -> None:
        definition = CapProgramDefinition(
            "ball_in_cup", "Pick the green ball, place it in the mug, verify, and return to rest "
            "in one reviewed program (needs SAM3; stops without closing if the ball is off-centre).")
        source = files("cap_harness.so101.programs").joinpath(f"{definition.name}.py").read_text()
        self._programs = {definition.name: LoadedCapProgram(definition, source, hashlib.sha256(source.encode()).hexdigest())}


class So101McpBridge(CapMcpBridge):
    def __init__(self, session: LiveAgentSession) -> None:
        super().__init__(session)
        self._direct = {
            "cap_" + spec.name.replace(".", "__"): spec
            for spec in session.registry.all_specs(public_only=True)
            if spec.name in DIRECT_TOOLS
        }

    def _dispatch(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if name == "observe":
            value = self._terminal(self.session.start_tool("get_observation", {}))
            return self._snapshot(value, args.get("cameras"))
        value, frames = super()._dispatch(name, args)
        if name in IMAGE_TOOLS:
            return value, frames
        return {**value, "image_cameras": []}, {}

    def _snapshot(self, value: dict[str, Any], cameras: Any = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        value, frames = super()._snapshot(value, cameras)
        if cameras is None and len(frames) > 1:
            height = min(frame.shape[0] for frame in frames.values())
            frames = {"+".join(frames): np.hstack([frame[:height] for frame in frames.values()])}
            value = {**value, "image_cameras": list(frames)}
        return value, frames


def create_so101_app(adapter: Any, *, event_sink: Any = None, tool_result_sink: Any = None) -> tuple[Any, Any]:
    """Return (FastAPI app with /mcp, /health and /steer, session); the caller owns the adapter."""
    from cap_harness.agent.server import create_app

    session = LiveAgentSession(So101SessionApi(adapter), adapter.register_tools(), program_catalog=So101Programs(),
                               event_sink=event_sink, tool_result_sink=tool_result_sink)
    return create_app(session, So101McpBridge), session
