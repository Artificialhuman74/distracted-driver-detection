#!/usr/bin/env bash
# Headless screenshot of the running demo: scripts/shoot.sh <out.png> <width> <height> [url]
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
P=$(mktemp -d); rm -f "$1"
"$C" --headless=new --disable-gpu --hide-scrollbars --user-data-dir="$P" --window-size="$2,$3" \
  --virtual-time-budget=12000 --screenshot="$1" "${4:-http://127.0.0.1:8000/}" >/dev/null 2>&1 &
PID=$!
for i in $(seq 1 40); do [ -s "$1" ] && sleep 1 && break; sleep 1; done
kill $PID 2>/dev/null; pkill -f "user-data-dir=$P" 2>/dev/null; rm -rf "$P"
ls -la "$1"
