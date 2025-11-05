#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Align-Then-Dash (Straight-Line Segments)
- State machine:
    ALIGN:   move sideways (vy only) until the target is centered (|theta| small)
    DASH:    move straight forward (vx only) toward the target
- If the target drifts off-center, return to ALIGN briefly, then DASH again.
- Pure translation (vx, vy); rotation locked (wz = 0.0)
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
# Alignment (sideways-only) behavior
VY_ALIGN_MAX        = 0.30      # max lateral speed while aligning
K_ALIGN             = 2.0       # tanh gain for mapping angle -> vy (vy = VY_ALIGN_MAX * tanh(K_ALIGN*|theta|))
TH_ALIGN            = 0.12      # radians; enter DASH when |theta| <= TH_ALIGN
TH_HYST             = 0.08      # extra margin to avoid chattering (return to ALIGN if |theta| > TH_ALIGN + TH_HYST)

# Dash (straight-in) behavior
VX_DASH_MAX         = 0.35      # top forward speed (m/s)
K_DASH              = 1.2       # distance shaping: vx = VX_DASH_MAX * tanh(K_DASH * d)

# Stop very near
NEAR_STOP_BAND_M    = 0.06      # if d < this, stop

# Vector acceleration limiting (keeps motion smooth without bending direction)
ACCEL_MAX           = 0.60      # m/s^2 (limit on change of the (vx,vy) vector magnitude)

# Loop / safety
LOOP_HZ             = 30
SCAN_TIMEOUT_S      = 0.20
FOLLOW_SIGN         = +1.0      # flip to -1.0 if your frame is reversed
PRINT_DEBUG         = True
# ====================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def find_min_nonzero(ranges, thetas):
    """Return (min_distance, angle_at_min) ignoring zero/invalid ranges. If none valid, (inf, 0)."""
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0.0
    if not np.any(mask):
        return (float("inf"), 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    idx = int(np.argmin(r_valid))
    return float(r_valid[idx]), float(t_valid[idx])

class Driver:
    """Abstracts MBot motion APIs; locks wz = 0 (no spin)."""
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
            self.mode = "set_velocity"   # (vx, wz)
        elif hasattr(bot, "motors"):
            self.mode = "motors"         # (left, right)
        else:
            raise RuntimeError("No valid drive function found on MBot")
        if PRINT_DEBUG:
            print(f"[Driver] Using mode: {self.mode}")

    def send(self, vx, vy=0.0, wz=0.0):
        wz = 0.0
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)                  # vy ignored on this platform
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, 0.0)      # no lateral channel
        elif self.mode == "motors":
            self.bot.motors(vx, vx)             # equal wheels ~ straight

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)

    # State machine
    ALIGN, DASH = 0, 1
    state = ALIGN

    last_scan_time = time.time()

    # Previous commanded velocity for vector accel limiting
    vx_prev, vy_prev = 0.0, 0.0

    print("[INFO] Follow-Me 2D (Align-Then-Dash) started. No spinning; straight-line segments toward the target.")

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
            if d < NEAR_STOP_BAND_M:
                # Close enough: stop but remain in current state
                v_des_x, v_des_y = 0.0, 0.0
            else:
                if state == ALIGN:
                    if th_abs <= TH_ALIGN:
                        state = DASH
                elif state == DASH:
                    if th_abs > TH_ALIGN + TH_HYST:
                        state = ALIGN

                # --------------- STATE ACTIONS ---------------
                if state == ALIGN:
                    # Sideways-only to center the target (no forward)
                    vy_mag = VY_ALIGN_MAX * math.tanh(K_ALIGN * th_abs)
                    vy_dir = 1.0 if th_min >= 0.0 else -1.0
                    v_des_x = 0.0
                    v_des_y = FOLLOW_SIGN * vy_mag * vy_dir

                else:  # DASH
                    # Straight-in: forward-only (no lateral)
                    vx_mag = VX_DASH_MAX * math.tanh(K_DASH * d)
                    v_des_x = FOLLOW_SIGN * vx_mag
                    v_des_y = 0.0

            # --------------- VECTOR ACCELERATION LIMITING ---------------
            dt = max(time.time() - t0, 1e-3)
            dvx = v_des_x - vx_prev
            dvy = v_des_y - vy_prev
            dv_norm = math.hypot(dvx, dvy)
            dv_max = ACCEL_MAX * dt
            if dv_norm > dv_max:
                scale = dv_max / dv_norm
                dvx *= scale
                dvy *= scale

            vx_cmd = vx_prev + dvx
            vy_cmd = vy_prev + dvy
            vx_prev, vy_prev = vx_cmd, vy_cmd

            # --------------- SEND (NO SPIN) ---------------
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] state={'ALIGN' if state==ALIGN else 'DASH'} | vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s | d={d:.3f} m, |theta|={th_abs:.3f} rad")

            # Keep loop timing
            loop_dt = time.time() - t0
            sleep_left = period - loop_dt
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

