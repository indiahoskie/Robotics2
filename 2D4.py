#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Straight-Line + Yaw Hold (No Turning)
- ALWAYS moves directly toward the nearest nonzero LiDAR return (attractor-only)
- Locks heading: maintains initial yaw using IMU-based feedback
- Pure translation (vx, vy); wz is used ONLY to cancel drift (commands ~0)
- Vector acceleration limiting to avoid jerks
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
# Speed shaping
V_MAX               = 0.30     # top translational speed (m/s) — conservative
V_MIN               = 0.00     # floor (0.0 = no creep when very small command)
K_SHAPE             = 1.0      # v = V_MAX * tanh(K_SHAPE * d)

# Vector acceleration limiting (applies to combined (vx,vy))
ACCEL_MAX           = 0.50     # m/s^2

# Stop very near
NEAR_STOP_BAND_M    = 0.07     # within this distance, stop

# Yaw hold (heading lock)
ENABLE_YAW_HOLD     = True
KYAW_HOLD           = 1.2      # rad/s per rad (small corrective gain)
WZ_HOLD_MAX         = 0.35     # cap on corrective wz (rad/s)

# Loop / safety
LOOP_HZ             = 30
SCAN_TIMEOUT_S      = 0.20
FOLLOW_SIGN         = +1.0     # flip to -1.0 if frame seems reversed
PRINT_DEBUG         = True
# ====================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ang_wrap(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2*math.pi) - math.pi

def ang_err(a, b):
    """Smallest signed difference a-b in [-pi, pi]."""
    return ang_wrap(a - b)

def find_min_nonzero(ranges, thetas):
    """
    Return (min_distance, angle_at_min) ignoring zero/invalid ranges.
    If none valid, returns (inf, 0.0).
    """
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0.0
    if not np.any(mask):
        return (float("inf"), 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    idx = int(np.argmin(r_valid))
    return float(r_valid[idx]), float(t_valid[idx])

class YawSensor:
    """Best-effort yaw reader across common MBot APIs."""
    def __init__(self, bot):
        self.bot = bot
        self.reader = None
        # Try common methods
        for name in ("read_imu", "readImu", "get_imu", "getImu", "imu"):
            if hasattr(bot, name):
                self.reader = getattr(bot, name)
                break

    def read_yaw(self):
        if self.reader is None:
            return None
        try:
            data = self.reader() if callable(self.reader) else self.reader
            # Accept tuple (roll,pitch,yaw)
            if isinstance(data, (list, tuple)) and len(data) >= 3:
                return float(data[2])
            # Accept dict-like {'yaw': ...}
            if isinstance(data, dict) and "yaw" in data:
                return float(data["yaw"])
            # Some APIs expose .yaw attribute
            if hasattr(data, "yaw"):
                return float(getattr(data, "yaw"))
        except Exception:
            return None
        return None

class Driver:
    """
    Motion adapter:
      - Prefers drive(vx, vy, wz)
      - Falls back gracefully if only vx (or vx,wz) exists
      - For motor-level, uses equal wheels to avoid rotation
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
        # We try to pass wz if supported; otherwise ensure no rotation via equal wheel speeds.
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            # No lateral channel; project onto x only (still translation; platform won't rotate here).
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)  # cannot command vy on this API
        elif self.mode == "motors":
            # Equal wheels => straight (no intentional rotation)
            self.bot.motors(vx, vx)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)
    imu = YawSensor(bot)

    # Heading lock
    yaw_ref = None

    last_scan_time = time.time()

    # Previous commanded velocity (for vector accel limiting)
    vx_prev, vy_prev = 0.0, 0.0

    print("[INFO] Follow-Me 2D (straight-line + yaw-hold) started. Translation only; heading locked.")

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

            # --------------- DESIRED TRANSLATIONAL VELOCITY (NO SMOOTHING) ---------------
            d = max(d_min, 0.0)
            if d < NEAR_STOP_BAND_M:
                v_des_x, v_des_y = 0.0, 0.0
            else:
                v_mag = V_MAX * math.tanh(K_SHAPE * d)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)
                ux = math.cos(th_min)
                uy = math.sin(th_min)
                v_des_x = FOLLOW_SIGN * v_mag * ux
                v_des_y = FOLLOW_SIGN * v_mag * uy

            # --------------- VECTOR ACCEL LIMIT ----------------
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

            # --------------- YAW HOLD (HEADING LOCK) ---------------
            wz_cmd = 0.0
            if ENABLE_YAW_HOLD:
                yaw = imu.read_yaw()
                if yaw is not None:
                    if yaw_ref is None:
                        yaw_ref = yaw  # set reference on first valid read
                    err = ang_err(yaw_ref, yaw)  # positive if yaw < ref (needs positive wz)
                    wz_cmd = clamp(KYAW_HOLD * err, -WZ_HOLD_MAX, +WZ_HOLD_MAX)
                # If no IMU, we keep wz_cmd = 0.0 and rely on equal-wheel logic

            # --------------- SEND ---------------
            driver.send(vx_cmd, vy_cmd, wz_cmd)

            if PRINT_DEBUG:
                v_cmd_mag = math.hypot(vx_cmd, vy_cmd)
                dbg = f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz={wz_cmd:+.3f} rad/s | d={d:.3f} m"
                if yaw_ref is not None and ENABLE_YAW_HOLD:
                    dbg += f", yaw_ref set"
                print(dbg)

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


