#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Calm & Straight-Line Segments
- Uses your original structure and Driver
- ALIGN: lateral-only to center target (vy, vx=0)
- DASH:  forward-only straight run (vx, vy=0)
- No spinning (wz = 0). Per-axis accel limit kept, but only one axis moves at a time.
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG (calmer tuning) ==============================
# --- (kept from your code where useful) ---
V_MAX               = 0.35     # (used as VX_DASH_MAX below)
V_MIN               = 0.02
K_SHAPE             = 1.2

# Direction smoothing & rate limiting (NOT used in straight-line mode)
SMOOTH_ALPHA_CALM   = 0.50
SMOOTH_ALPHA_AGGRO  = 0.85
AGGRO_DIST_THRESH_M = 0.30
THETA_RATE_MAX      = 2.0

# Acceleration limiting (per axis)
ACCEL_MAX           = 0.60     # m/s^2
NEAR_STOP_BAND_M    = 0.06

# Loop / safety
LOOP_HZ             = 30
SCAN_TIMEOUT_S      = 0.20
FOLLOW_SIGN         = +1.0
PRINT_DEBUG         = True

# ==================== NEW: Straight-line state machine tuning ======================
# ALIGN (sideways) until target is centered:
VY_ALIGN_MAX        = 0.30                 # max lateral speed during ALIGN
K_ALIGN             = 2.0                  # vy = VY_ALIGN_MAX * tanh(K_ALIGN * |theta|)
TH_ALIGN            = 0.12                 # rad; enter DASH when |theta| <= TH_ALIGN
TH_HYST             = 0.08                 # hysteresis to avoid chattering

# DASH (forward-only) toward target:
VX_DASH_MAX         = V_MAX                # reuse your V_MAX
K_DASH              = K_SHAPE              # vx = VX_DASH_MAX * tanh(K_DASH * d)
# ====================================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ang_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi

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

    last_scan_time = time.time()

    # Per-axis accel limiting memory
    vx_prev = 0.0
    vy_prev = 0.0

    # --- Straight-line state machine ---
    ALIGN, DASH = 0, 1
    state = ALIGN

    print("[INFO] Follow-Me 2D: Straight-line segments (ALIGN -> DASH). No spinning.")

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
                    print(f"[SCAN] min_dist={d_min:.3f} m at theta={th_min:+.3f} rad | state={'ALIGN' if state==ALIGN else 'DASH'}")
                else:
                    print("[SCAN] No valid returns.")

            if not np.isfinite(d_min):
                driver.send(0.0, 0.0, 0.0)
                time.sleep(period)
                continue

            # --------------- STATE TRANSITIONS ---------------
            th_abs = abs(th_min)
            d = max(d_min, 0.0)

            # default desired velocities
            v_des_x, v_des_y = 0.0, 0.0

            if d < NEAR_STOP_BAND_M:
                # Close: stop but keep state
                v_des_x, v_des_y = 0.0, 0.0
            else:
                if state == ALIGN:
                    # If centered enough, switch to DASH
                    if th_abs <= TH_ALIGN:
                        state = DASH
                else:  # state == DASH
                    # If target drifts off-center, go back to ALIGN
                    if th_abs > TH_ALIGN + TH_HYST:
                        state = ALIGN

                # --------------- STATE ACTIONS ---------------
                if state == ALIGN:
                    # Lateral-only (straight sideways). vx=0, vy!=0
                    vy_mag = VY_ALIGN_MAX * math.tanh(K_ALIGN * th_abs)
                    vy_dir = 1.0 if th_min >= 0.0 else -1.0
                    v_des_x = 0.0
                    v_des_y = FOLLOW_SIGN * vy_mag * vy_dir

                elif state == DASH:
                    # Forward-only straight run. vy=0, vx!=0
                    vx_mag = VX_DASH_MAX * math.tanh(K_DASH * d)
                    # apply floor if needed
                    if vx_mag > 0.0:
                        vx_mag = max(vx_mag, V_MIN)
                    v_des_x = FOLLOW_SIGN * vx_mag
                    v_des_y = 0.0

            # --------------- PER-AXIS ACCEL LIMITING ---------------
            dt = max(time.time() - t0, 1e-3)
            dv_max = ACCEL_MAX * dt

            vx_cmd = clamp(v_des_x - vx_prev, -dv_max, +dv_max) + vx_prev
            vy_cmd = clamp(v_des_y - vy_prev, -dv_max, +dv_max) + vy_prev

            vx_prev, vy_prev = vx_cmd, vy_cmd

            # --------------- SEND (NO SPIN) ---------------
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] {('ALIGN' if state==ALIGN else 'DASH'):>5} | vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s | d={d:.3f} m, |θ|={th_abs:.3f} rad")

            # Maintain loop period
            dt_loop = time.time() - t0
            sleep_left = period - dt_loop
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
