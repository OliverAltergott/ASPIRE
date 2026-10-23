#!/usr/bin/env bash
# Attach a fresh Codex agent to a running SO101 MCP runtime (python -m cap_harness.so101 --mcp).
# Usage: scripts/start_so101_agent.sh [exec] [task text]
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASPIRE_PORT="${ASPIRE_MCP_PORT:-8222}"
[[ "$ASPIRE_PORT" =~ ^[0-9]+$ ]] || { printf 'ASPIRE_MCP_PORT must be numeric.\n' >&2; exit 1; }
curl --noproxy '*' --fail --silent "http://127.0.0.1:$ASPIRE_PORT/health" >/dev/null \
  || { printf 'Start the SO101 MCP runtime first.\n' >&2; exit 1; }
ASPIRE_GUIDE="$(cat "$ASPIRE_ROOT/docs/so101-agent-guide.md")"
ASPIRE_PERMISSIONS=()
for ASPIRE_TOOL in get_capability_graph observe get_job read_operator_messages cap_get_robot_state \
  cap_move_to_joints cap_set_gripper cap_so101__enable cap_so101__get_telemetry cap_so101__move_delta \
  cap_so101__move_tool cap_so101__go_to cap_so101__close_until_contact cap_so101__detect_color \
  cap_so101__segment_text cap_program_ball_in_cup; do
  ASPIRE_PERMISSIONS+=(-c "mcp_servers.cap.tools.$ASPIRE_TOOL.approval_mode=\"approve\"")
done
ASPIRE_MODE=()
if [[ "${1:-}" == "exec" ]]; then ASPIRE_MODE=(exec); shift; fi
ASPIRE_TASK="${*:-After startup completes, wait for the operator task.}"
exec codex "${ASPIRE_MODE[@]}" -C "$ASPIRE_ROOT" \
  -c 'approval_policy="never"' \
  -c 'sandbox_mode="workspace-write"' \
  -c 'sandbox_workspace_write.network_access=true' \
  -c "mcp_servers.cap.url=\"http://127.0.0.1:$ASPIRE_PORT/mcp/\"" \
  -c 'mcp_servers.cap.enabled=true' -c 'mcp_servers.cap.required=true' \
  -c 'mcp_servers.cap.tool_timeout_sec=240' \
  "${ASPIRE_PERMISSIONS[@]}" \
  "$ASPIRE_GUIDE
Operator task:
$ASPIRE_TASK"
