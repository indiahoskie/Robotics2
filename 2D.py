#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR)
- Finds the nearest obstacle (ignoring zero ranges)
- Moves the robot laterally toward that obstacle's direction
- Maintains a setpoint distance using P-control
- Does NOT spin: wz is held at 0.0 (pure translation in x/y)

Expected behavior:
- Robot follows the nearest obstacle/human as they move around
- Can move in multiple directions in the plane (2D) without rotating to face the target
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------------------- CONFIG ----------------------------
SETPOINT_M      = 0.75     # desired distance to target (m)
TOLERANCE_M     = 0.05     # acceptable error band around setpoint (m)

KP_DIST         = 1.0      # P gain for radial distance control (m/s per meter error)

V_MAX           = 0.50     # absolute cap on translational speed (m/s)
V_MIN           = 0.05     # small floor to overcome static friction when moving (m/s)
DEADBAND_M      = 0.02     # small deadband near zero error
SMOOTH_ALPHA    = 0.35     # EMA smoothing for theta_min (0=no smoothing, 1=full overwrite)

SCAN_TIMEOUT_S  = 0.25     # if scans lag beyond this, stop for safety
LOOP_HZ         = 15       # control loop rate

PRINT_DEBUG     = True
# -----------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ema(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev

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
    idx = np.argmin(r_valid)
    return float(r_valid[idx]), float(t_valid[idx])

class Driver:
    """
    Motion adapter similar to the 1D script, but prefers full planar control:
    - Preferred: drive(vx, vy, wz)
    - Fallbacks keep API compatible; wz stays 0 to avoid spinning
    """
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None

        if hasattr(bot, "drive"):
            sig = inspect.signature(bot.drive)
            # choose the most capable signature first
            if len(sig.parameters) >= 3:
                self.mode = "drive_vx_vy_wz"  # ideal for 2D translation
            elif len(sig.parameters) == 1:
                self.mode = "drive_vx"        # limited; no vy, no wz
            else:
                self.mode = "drive_generic"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"        # set_velocity(vx, wz) -- not ideal for 2D
        elif hasattr(bot, "motors"):
            self.mode = "motors"              # very low-level differential approx
        else:
            raise RuntimeError("No valid drive function found on MBot")

        if PRINT_DEBUG:
            print(f"[Driver] Using mode: {self.mode}")

    def send(self, vx, vy=0.0, wz=0.0):
        # lock rotation to zero (requirement: do not spin)
        wz = 0.0

        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx":
            # No vy available; approximate by projecting into x only.
            # Robot won’t truly strafe; this is a compatibility fallback.
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            # set_velocity(vx, wz) — cannot command vy; keep wz=0
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
            # Differential fallback: approximate vx with equal wheel speeds.
            # No strafing capability; still avoids rotation.
            left = vx
            right = vx
            self.bot.motors(left, right)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)

    # Smoothed theta for stable direction
    theta_smooth = 0.0
    last_scan_time = time.time()

    print("[INFO] Follow-Me 2D started (no spinning). Maintaining setpoint distance to nearest obstacle.")

    try:
        while True:
            # ---- READ LIDAR ----
            # Expect: ranges, thetas where thetas are radians in robot frame
            # (front ~ 0, left ~ +pi/2, right ~ -pi/2, behind ~ +/- pi)
            ranges, thetas = bot.read_lidar()
            now = time.time()
            if now - last_scan_time > SCAN_TIMEOUT_S:
                # If scans are laggy or missing, stop for safety
                driver.send(0.0, 0.0, 0.0)
            last_scan_time = now

            # ---- FIND NEAREST VALID RETURN ----
            d_min, th_min = find_min_nonzero(ranges, thetas)

            if PRINT_DEBUG:
                if np.isfinite(d_min):
                    print(f"[SCAN] min_dist={d_min:.3f} m at theta={th_min:+.3f} rad")
                else:
                    print("[SCAN] No valid returns (all zero/invalid).")

            if not np.isfinite(d_min):
                # No target: gently stop
                driver.send(0.0, 0.0, 0.0)
                time.sleep(1.0 / LOOP_HZ)
                continue

            # ---- SMOOTH DIRECTION ----
            theta_smooth = ema(theta_smooth, th_min, SMOOTH_ALPHA)

            # ---- RADIAL P-CONTROL ON DISTANCE ----
            # Positive error => move toward target; Negative => back away
            err = d_min - SETPOINT_M

            # Deadband near target to avoid jitter
            if abs(err) < DEADBAND_M:
                vx_cmd = 0.0
                vy_cmd = 0.0
            else:
                # Speed magnitude from P-control
                v_mag = KP_DIST * err

                # Clamp and apply a small floor so it actually moves when needed
                v_sign = 1.0 if v_mag >= 0 else -1.0
                v_mag = abs(v_mag)
                v_mag = clamp(v_mag, 0.0, V_MAX)
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)

                # Convert polar (v_mag, theta_smooth) -> Cartesian (vx, vy)
                # Robot frame convention: theta=0 forward, +left, -right.
                vx_cmd = v_mag * math.cos(theta_smooth)
                vy_cmd = v_mag * math.sin(theta_smooth)

            # ---- SEND COMMANDS (NO SPIN) ----
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz=+0.000 rad/s | err={err:+.3f} m")

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()

