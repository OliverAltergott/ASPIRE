# SO101 agent operating guide

These instructions are for a coding agent operating an SO101 through the CAP MCP session
served by `python -m cap_harness.so101 --mcp`. `scripts/start_so101_agent.sh` includes
this file in the agent's initial instructions.

This station is a single five-joint SO101 with two RGB cameras, not a YAM. In
[AGENTS.md](../AGENTS.md), the execution, evidence, recovery and completion rules apply;
its YAM startup Home, `cap_program_go_home` and the YAM `cap-tool-usage` skill do not.
Read [SO101](so101.md) once for the tools, interlocks and frame conventions, and the
matching task recipe, e.g. [ball into a cup](../examples/so101/ball_in_cup/TASK_CONTEXT.md).

Startup: call `observe`, then `cap_so101__enable`. Enable never moves the arm; it holds
the current pose, which may be the folded rest pose. Report `READY` with the measured
joints and tool position from `cap_so101__get_telemetry`.

- `observe` returns both cameras as one side-by-side image (external | wrist) and is the
  inspection that motion requires within 60 s. Motion results return no images, so look
  when a decision depends on it. Images per request are capped, so do not observe after
  every step.
- Between looks, measure: `cap_so101__segment_text` (SAM3; about 5 s per camera) or
  `cap_so101__detect_color` give pixel positions, and telemetry gives the URDF tool
  position. Neither camera has depth or extrinsics.
- For the ball-into-cup task, prefer the reviewed program `cap_program_ball_in_cup`: one
  call does the whole task and fails if the ball ends visibly outside the mug. If it
  stops (for example, the ball is outside the grasp window), observe, then continue with
  the individual tools following the recipe.
- Call tools one at a time. A refused call wrote nothing; read its message and adjust.
- Finish by placing any held object, then `cap_so101__go_to` with the rest joints from
  the recipe. Torque stays on when the session ends.
