#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Attractor-Only
- ALWAYS moves toward the nearest nonzero LiDAR return
- No setpoint / no "back away" logic
- Pure translation in the plane (vx, vy); rotation locked (wz = 0.0)

Expected behavior:
- As you move around the robot, it translates to follow you and close the distance.
- If your MBot frame is flipped (it runs away), set FOLLOW_SIGN = -1.0.
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
# Motion scaling
KP_TOWARD           = 2.0     # How strongly speed scales with distance (m/s per m)
V_MAX               = 0.80    # Max translational speed magnitude (m/s)
V_MIN               = 0.05    # Small floor to overcome static friction when moving

# Direction smoothing (less when you're far/moving, more when stable)
SMOOTH_ALPHA_CALM   = 0.60    # keep more of past angle when near/stable
SMOOTH_ALPHA_AGGRO  = 0.95    # update direction aggressively when far/moving
AGGRO_DIST_THRESH_M = 0.25    # above this distance -> use aggressive smoothing

# Safety / loop
SCAN_TIMEOUT_S      = 0.15    # if scans stall longer than this, stop
LOOP_HZ             = 40      # control loop rate (Hz)

# Behavior toggles
FOLLOW_SIGN         = +1.0    # if robot runs away, flip to -1.0
NEAR_STOP_BAND_M    = 0.03    # if target is extremely close, stop to avoid bumping

PRINT_DEBUG         = True
# ====================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ang_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def ang_lerp(prev, new, alpha):
    """Interpolate angles on the circle: alpha in (0..1] toward 'new'."""
    return prev + ang_diff(new, prev) * alpha

def find_min_nonzero(ranges, thetas):
    """
    Return (min_distance, angle_at_min) ignoring zero/invalid ranges.
    If none valid, returns (np.inf, 0.0).
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
      - Prefers drive(vx, vy, wz)
      - Falls back gracefully when only vx (or vx,wz) is available
      - Rotation (wz) is forced to 0.0 (no spinning)
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
            self.mode = "set_velocity"  # (vx, wz)
        elif hasattr(bot, "motors"):
            self.mode = "motors"        # (left, right)
        else:
            raise RuntimeError("No valid drive function found on MBot")

        if PRINT_DEBUG:
            print(f"[Driver] Using mode: {self.mode}")

    def send(self, vx, vy=0.0, wz=0.0):
        wz = 0.0  # lock rotation (no spin)
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            # No lateral channel; project onto x only (still approaches when in front/behind).
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            # API: set_velocity(vx, wz). Keep wz=0.
            self.bot.set_velocity(vx, 0.0)
        elif self.mode == "motors":
            # Equal wheels approximate straight translation (no spin).
            self.bot.motors(vx, vx)

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

    print("[INFO] Follow-Me 2D (attractor-only) started. No spinning; always moves toward nearest obstacle.")

    try:
        period = 1.0 / LOOP_HZ
        while True:
            # ------------------ READ LIDAR ------------------
            # Expect: ranges (m), thetas (rad) in robot frame:
            #   front ~ 0, left ~ +pi/2, right ~ -pi/2, behind ~ +/-pi
            ranges, thetas = bot.read_lidar()

            now = time.time()
            if now - last_scan_time > SCAN_TIMEOUT_S:
                # Scans lagging: stop for safety
                driver.send(0.0, 0.0, 0.0)
            last_scan_time = now

            # --------------- NEAREST VALID RETURN ---------------
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

            # --------------- ADAPTIVE DIRECTION SMOOTHING ---------------
            # When target is far/moving, turn the velocity direction faster (less smoothing).
            alpha = SMOOTH_ALPHA_AGGRO if d_min > AGGRO_DIST_THRESH_M else SMOOTH_ALPHA_CALM
            theta_smooth = ang_lerp(theta_smooth, th_min, alpha)

            # --------------- ATTRACTOR-ONLY TRANSLATION ---------------
            # Always move *toward* the target. Speed grows with distance.
            d = max(d_min, 0.0)

            if d < NEAR_STOP_BAND_M:
                # Extremely close: stop to avoid bumping
                vx_cmd = 0.0
                vy_cmd = 0.0
            else:
                v_mag = KP_TOWARD * d
                v_mag = clamp(v_mag, 0.0, V_MAX)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)

                # Convert polar (toward theta_smooth) to Cartesian
                vx_cmd = FOLLOW_SIGN * v_mag * math.cos(theta_smooth)
                vy_cmd = FOLLOW_SIGN * v_mag * math.sin(theta_smooth)

            # --------------- SEND (NO SPIN) ---------------
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz=+0.000 | d={d:.3f} m (alpha={alpha:.2f})")

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
