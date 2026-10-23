#!/usr/bin/env bash
# Run the reviewed SO101 ball-into-cup program once, recorded, without an agent.
#   SO101_PORT=/dev/cu.usbmodemXXXX SO101_ROBOT_ID=my_follower scripts/so101_ball_in_cup.sh
# Starts the portable SAM3 service if it is not already running.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SO101_PYTHON:-$ROOT/.venv-so101/bin/python}"
: "${SO101_PORT:?set SO101_PORT, e.g. /dev/cu.usbmodemXXXX}"
: "${SO101_ROBOT_ID:?set SO101_ROBOT_ID}"
SO101_CALIBRATION_DIR="${SO101_CALIBRATION_DIR:-$HOME/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
SO101_CAMERAS="${SO101_CAMERAS:-{\"external\": 1, \"wrist\": 0}}"
SO101_OUTPUT="${SO101_OUTPUT:-$ROOT/outputs/so101}"
SAM3_PORT="${SAM3_PORT:-8114}"
mkdir -p "$SO101_OUTPUT"

if ! curl --noproxy '*' --fail --silent "http://127.0.0.1:$SAM3_PORT/healthz" >/dev/null; then
  (cd /tmp && "$PY" -m cap_harness.providers.sam3.portable --device "${SAM3_DEVICE:-cpu}" --port "$SAM3_PORT" \
     >"$SO101_OUTPUT/sam3-service.log" 2>&1 &)
  for _ in $(seq 1 120); do
    curl --noproxy '*' --fail --silent "http://127.0.0.1:$SAM3_PORT/healthz" >/dev/null && break
    sleep 1
  done
fi

START=$(date +%s)
STATUS=0
(cd /tmp && "$PY" -m cap_harness.so101 --port "$SO101_PORT" --robot-id "$SO101_ROBOT_ID" \
  --calibration-dir "$SO101_CALIBRATION_DIR" --cameras "$SO101_CAMERAS" --output "$SO101_OUTPUT" \
  --allow-motion --speed "${SO101_SPEED:-60}" --record --sam3-url "http://127.0.0.1:$SAM3_PORT" \
  --program "$ROOT/src/cap_harness/so101/programs/ball_in_cup.py") || STATUS=$?
printf 'program finished in %s s (exit %s)\n' "$(( $(date +%s) - START ))" "$STATUS"
exit "$STATUS"
