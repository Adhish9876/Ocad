# OralScan — Raspberry Pi 5 Setup Guide

## Hardware Requirements
- Raspberry Pi 5
- Raspberry Pi Camera v2 (connected via CSI ribbon)
- ULN2003 stepper motor driver board + 28BYJ-48 motor
- GPIO wiring:
  | Driver Pin | Pi GPIO (BCM) |
  |---|---|
  | IN1 | GPIO 17 |
  | IN2 | GPIO 18 |
  | IN3 | GPIO 27 |
  | IN4 | GPIO 22 |
  | GND | Pi GND |
  | VCC | 5V (external supply) |

## One-Time Setup (on Pi)

```bash
# 1. Copy the project folder to the Pi
#    (e.g. via scp, USB drive, or git clone)

# 2. Install dependencies
pip install -r requirements_pi.txt

# 3. Make launch script executable
chmod +x start.sh
```

## Launch (every time)

```bash
bash start.sh
```

That's it! Chromium opens in full-screen kiosk mode at `http://localhost:5000`.

## Using the App

1. **Upload tab** — works just like before: upload strip photos to get prognosis
2. **Live Scanner tab** — click **Start Scan**:
   - App calibrates camera + motor (short spin test)
   - Motor starts spinning the belt/turntable
   - When a strip is detected (3 consecutive confirmed frames), motor stops automatically
   - A cropped photo is saved to `~/smd_captures/`
   - Motor resumes immediately
   - Click **Analyze** on any captured image to run YOLOv8 and see prognosis
3. **Stop** button stops the motor and scan loop at any time

## Auto-start on Boot (optional)

Add this line to `/etc/rc.local` (before `exit 0`):

```bash
su - pi -c "bash /home/pi/your-app-folder/start.sh &"
```

## Captured Images

All captured strip images are saved to `~/smd_captures/` on the Pi.
They persist between sessions and can be re-analyzed at any time from the gallery.

## Troubleshooting

| Issue | Fix |
|---|---|
| Camera not found | Run `libcamera-hello` to test camera |
| Motor not spinning | Check GPIO wiring and ensure 5V external power |
| App doesn't load | Check `python app.py` is running (`ps aux | grep app.py`) |
| Chromium won't open | Try `chromium-browser http://localhost:5000` manually |
