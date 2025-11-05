#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Fast Response
- Tracks the nearest nonzero range and translates toward it in the plane
- Adaptive smoothing for direction (less smoothing when you're moving)
- Higher loop rate, higher P gain, smaller deadband
- Still NO rotation: wz = 0.0
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------------------- CONFIG (tuned for snappier feel) ----------------------------
SETPOINT_M      = 0.75     # desired distance to target (m)
TOLERANCE_M     = 0.05

KP_DIST         = 1.8      # ↑ faster radial response (was 1.0)

V_MAX           = 0.80     # ↑ allow quicker bursts
V_MIN           = 0.05
DEADBAND_M      = 0.01     # ↓ less "no-move" zone near setpoint

# Direction smoothing (adaptive): more smoothing when calm, less when moving
SMOOTH_ALPHA_CALM   = 0.55   # how much of the NEW theta to keep when calm
SMOOTH_ALPHA_AGGRO  = 0.95   # when you're moving, almost no smoothing
AGGRO_ERR_THRESH_M  = 0.12   # err above this => aggressive direction updates

SCAN_TIMEOUT_S  = 0.15     # ↓ stop sooner if scans stall
LOOP_HZ         = 40       # ↑ faster control loop
PRINT_DEBUG     = True
# ------------------------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

def ang_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return d

def ang_lerp(prev, new, alpha):
    """Interpolate angles correctly on the circle."""
    return prev + ang_diff(new, prev) * alpha

def find_min_nonzero(ranges, thetas):
    """
    Return (min_distance, angle_at_min) ignoring zero/invalid ranges.
    If nothing valid, returns (np.inf, 0.0)
    """
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0.0
    if not np.any(mask):
        return (np.inf, 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    idx = int(np.argmin(r_valid))
    return float(r_valid[idx]), float(t_valid[idx])

class Driver:
    """
    Motion adapter:
    - Prefers drive(vx, vy, wz). Falls back gracefully. wz forced to 0.0.
    """
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None

        if hasattr(bot, "drive"):
            sig = inspect.signature(bot.drive)
            if len(sig.parameters) >= 3:
                self.mode = "drive_vx_vy_wz"
            elif len(sig.parameters) == 1:
                self.mode = "drive_vx"
            else:
                self.mode = "drive_generic"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"
        elif hasattr(bot, "motors"):
            self.mode = "motors"
        else:
            raise RuntimeError("No valid drive function found on MBot")

        if PRINT_DEBUG:
            print(f"[Driver] Using mode: {self.mode}")

    def send(self, vx, vy=0.0, wz=0.0):
        wz = 0.0  # lock rotation
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)            # no lateral ability; still follows forward/backward projection
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz) # cannot command vy on this API
        elif self.mode == "motors":
            self.bot.motors(vx, vx)       # equal wheels => pure translation (approx.)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)

    theta_smooth = 0.0
    last_scan_time = time.time()

    print("[INFO] Follow-Me 2D (fast) started. No spinning; maintaining setpoint to nearest obstacle.")

    try:
        period = 1.0 / LOOP_HZ
        while True:
            # ---- READ LIDAR ----
            ranges, thetas = bot.read_lidar()
            now = time.time()
            if now - last_scan_time > SCAN_TIMEOUT_S:
                driver.send(0.0, 0.0, 0.0)
            last_scan_time = now

            # ---- FIND NEAREST VALID RETURN ----
            d_min, th_min = find_min_nonzero(ranges, thetas)

            if PRINT_DEBUG:
                if np.isfinite(d_min):
                    print(f"[SCAN] min_dist={d_min:.3f} m at theta={th_min:+.3f} rad")
                else:
                    print("[SCAN] No valid returns.")

            if not np.isfinite(d_min):
                driver.send(0.0, 0.0, 0.0)
                time.sleep(period)
                continue

            # ---- ADAPTIVE DIRECTION SMOOTHING ----
            err = d_min - SETPOINT_M
            # More error => move direction faster (less smoothing)
            alpha = SMOOTH_ALPHA_AGGRO if abs(err) > AGGRO_ERR_THRESH_M else SMOOTH_ALPHA_CALM
            theta_smooth = ang_lerp(theta_smooth, th_min, alpha)

            # ---- RADIAL P-CONTROL ----
            if abs(err) < DEADBAND_M:
                vx_cmd = 0.0
                vy_cmd = 0.0
            else:
                v_mag = KP_DIST * err   # toward target if positive, away if negative
                sgn = 1.0 if v_mag >= 0 else -1.0
                v_mag = clamp(abs(v_mag), 0.0, V_MAX)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)
                vx_cmd = v_mag * math.cos(theta_smooth) * sgn
                vy_cmd = v_mag * math.sin(theta_smooth) * sgn

            # ---- SEND (NO SPIN) ----
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz=+0.000 | err={err:+.3f} m (alpha={alpha:.2f})")

            time.sleep(period)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()
