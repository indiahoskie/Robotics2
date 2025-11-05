#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Calm & Smooth
- Always moves toward the nearest nonzero LiDAR return (attractor-only)
- Smooth speed shaping (tanh), acceleration limiting, and gentle direction updates
- Pure translation in the plane (vx, vy); rotation locked (wz = 0.0)
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG (calmer tuning) ==============================
# Speed shaping (so it doesn't surge)
V_MAX               = 0.35     # lower top speed (m/s)
V_MIN               = 0.02     # tiny floor to overcome static friction (can set 0.0 if you want zero creep)
K_SHAPE             = 1.2      # tanh shaping gain: v ≈ V_MAX * tanh(K_SHAPE * d)

# Direction smoothing & rate limiting
SMOOTH_ALPHA_CALM   = 0.50     # more smoothing when close
SMOOTH_ALPHA_AGGRO  = 0.85     # still some smoothing when far
AGGRO_DIST_THRESH_M = 0.30     # above this, use aggro smoothing
THETA_RATE_MAX      = 2.0      # rad/s max change in desired direction (prevents quick whips)

# Acceleration limiting (per axis)
ACCEL_MAX           = 0.60     # m/s^2; per-axis accel cap (vx & vy)
NEAR_STOP_BAND_M    = 0.06     # if extremely close, stop to avoid bumping

# Loop / safety
LOOP_HZ             = 30
SCAN_TIMEOUT_S      = 0.20
FOLLOW_SIGN         = +1.0     # flip to -1.0 if your frame is reversed
PRINT_DEBUG         = True
# ====================================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ang_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def ang_lerp(prev, new, alpha):
    """Interpolate angles on the circle: alpha in (0..1] toward 'new'."""
    return prev + ang_diff(new, prev) * alpha

def ang_rate_limit(prev, new, max_rate_per_sec, dt):
    """Limit the change between angles to max_rate_per_sec * dt."""
    d = ang_diff(new, prev)
    max_step = max_rate_per_sec * max(dt, 1e-3)
    d = clamp(d, -max_step, +max_step)
    return prev + d

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
        wz = 0.0  # lock rotation
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)            # no vy channel; projects onto x only
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, 0.0)
        elif self.mode == "motors":
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

    # For accel limiting
    vx_prev = 0.0
    vy_prev = 0.0

    print("[INFO] Follow-Me 2D (calm) started. No spinning; smooth translation toward nearest obstacle.")

    try:
        period = 1.0 / LOOP_HZ
        while True:
            t0 = time.time()
            # ------------------ READ LIDAR ------------------
            ranges, thetas = bot.read_lidar()

            now = time.time()
            if now - last_scan_time > SCAN_TIMEOUT_S:
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

            # --------------- DIRECTION (SMOOTH + RATE-LIMITED) ---------------
            alpha = SMOOTH_ALPHA_AGGRO if d_min > AGGRO_DIST_THRESH_M else SMOOTH_ALPHA_CALM
            theta_target = ang_lerp(theta_smooth, th_min, alpha)
            theta_smooth = ang_rate_limit(theta_smooth, theta_target, THETA_RATE_MAX, period)

            # --------------- SPEED SHAPING (TANH) ---------------
            d = max(d_min, 0.0)
            if d < NEAR_STOP_BAND_M:
                v_mag = 0.0
            else:
                # Smoothly increases with distance; saturates near V_MAX.
                v_mag = V_MAX * math.tanh(K_SHAPE * d)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)

            # Convert to x/y in robot frame (no rotation)
            vx_des = FOLLOW_SIGN * v_mag * math.cos(theta_smooth)
            vy_des = FOLLOW_SIGN * v_mag * math.sin(theta_smooth)

            # --------------- ACCELERATION LIMITING (per axis) ---------------
            dv_max = ACCEL_MAX * period
            vx_cmd = clamp(vx_des - vx_prev, -dv_max, +dv_max) + vx_prev
            vy_cmd = clamp(vy_des - vy_prev, -dv_max, +dv_max) + vy_prev

            # Save for next cycle
            vx_prev, vy_prev = vx_cmd, vy_cmd

            # --------------- SEND (NO SPIN) ---------------
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz=+0.000 | d={d:.3f} m | v_mag={v_mag:.3f}")

            # Maintain loop period
            dt = time.time() - t0
            sleep_left = period - dt
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()
