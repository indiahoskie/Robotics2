#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Straight-Line, Calm
- Always moves directly toward the nearest nonzero LiDAR return (attractor-only)
- No angle smoothing or rate limiting (avoids curved paths)
- Vector acceleration limiting to keep direction straight while smoothing motion
- Pure translation (vx, vy); rotation locked (wz = 0.0)
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
# Speed shaping (gentle, but decisive)
V_MAX               = 0.35     # top speed (m/s)
V_MIN               = 0.02     # floor to overcome static friction (set 0.0 for no creep)
K_SHAPE             = 1.2      # v_mag = V_MAX * tanh(K_SHAPE * d)

# Vector acceleration limiting (applies to the whole (vx,vy) vector)
ACCEL_MAX           = 0.60     # m/s^2 (limit on the change of speed vector magnitude)

# Stop very near
NEAR_STOP_BAND_M    = 0.06     # within this distance, stop

# Loop / safety
LOOP_HZ             = 30
SCAN_TIMEOUT_S      = 0.20
FOLLOW_SIGN         = +1.0     # flip to -1.0 if your frame is reversed
PRINT_DEBUG         = True
# ====================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

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
      - Falls back gracefully if only vx (or vx,wz) exists
      - wz forced to 0.0 (no spin)
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
        wz = 0.0
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

    # Previous commanded velocity (for vector accel limiting)
    vx_prev, vy_prev = 0.0, 0.0

    print("[INFO] Follow-Me 2D (straight-line) started. No spinning; moves directly toward nearest obstacle.")

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

            # --------------- DIRECT DIRECTION (NO SMOOTHING) ---------------
            # Always point the velocity exactly toward the nearest return.
            d = max(d_min, 0.0)
            if d < NEAR_STOP_BAND_M:
                v_des_x, v_des_y = 0.0, 0.0
            else:
                # Speed shaping: smooth growth with distance; saturates to V_MAX
                v_mag = V_MAX * math.tanh(K_SHAPE * d)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)
                # Unit direction straight at target (no lag)
                ux = math.cos(th_min)
                uy = math.sin(th_min)
                v_des_x = FOLLOW_SIGN * v_mag * ux
                v_des_y = FOLLOW_SIGN * v_mag * uy

            # --------------- VECTOR ACCELERATION LIMITING ---------------
            # Limit change in the velocity *vector* to ACCEL_MAX * dt
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

            # Save for next cycle
            vx_prev, vy_prev = vx_cmd, vy_cmd

            # --------------- SEND (NO SPIN) ---------------
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                v_cmd_mag = math.hypot(vx_cmd, vy_cmd)
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, |v|={v_cmd_mag:.3f} | d={d:.3f} m")

            # Maintain loop period
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
