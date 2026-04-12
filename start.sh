#!/bin/bash
# OralScan + SMD Scanner — One-click launcher for Raspberry Pi 5
# Usage: bash start.sh

cd "$(dirname "$0")"

echo "==================================================="
echo "  OralScan — Starting backend..."
echo "==================================================="

# Kill any previous instance
pkill -f "python app.py" 2>/dev/null
sleep 1

# Start Flask backend in background
python app.py &
FLASK_PID=$!

echo "[INFO] Flask PID: $FLASK_PID"
echo "[INFO] Waiting for server to be ready..."
sleep 4

# Open Chromium in kiosk mode
echo "[INFO] Launching Chromium kiosk..."
if command -v chromium-browser &> /dev/null; then
    BROWSER_CMD="chromium-browser"
else
    BROWSER_CMD="chromium"
fi

$BROWSER_CMD \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --app=http://localhost:5000

# When Chromium closes, also stop Flask
echo "[INFO] Chromium closed. Stopping Flask..."
kill $FLASK_PID 2>/dev/null
