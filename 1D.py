#!/usr/bin/env python3
"""
1D Follow-Me Controller using LiDAR
The robot moves forward or backward to maintain a set distance from an object in front.
"""

import time
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------- CONFIGURATION ----------------
SETPOINT = 0.6       # target distance (meters)
TOLERANCE = 0.05     # acceptable range around setpoint
KP = 1.2             # proportional gain
MAX_FWD = 0.35       # max forward speed (m/s)
MAX_REV = -0.30      # max reverse speed (m/s)
WINDOW = 8           # number of rays used at front
LOOP_HZ = 10         # control frequency (Hz)
# ------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def find_fwd_dist(ranges, thetas, window=WINDOW):
    """Compute the average forward distance based on front LiDAR readings."""
    if not ranges or not thetas:
        return np.inf

    fwd_ranges = np.array(ranges[:window] + ranges[-window:], dtype=float)
    fwd_thetas = np.array(thetas[:window] + thetas[-window:], dtype=float)

    valid = fwd_ranges > 0
    if not np.any(valid):
        return np.inf

    fwd_dists = fwd_ranges[valid] * np.cos(fwd_thetas[valid])
    fwd_dists = fwd_dists[fwd_dists > 0]
    if fwd_dists.size == 0:
        return np.inf
    return float(np.mean(fwd_dists))

# --- Motion adapter: supports different robot drive methods ---
class Driver:
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None
        if hasattr(bot, "drive"):
            sig = inspect.signature(bot.drive)
            if len(sig.parameters) >= 3:
                self.mode = "drive_vx_vy_wz"
            else:
                self.mode = "drive_vx"
        elif hasattr(bot, "motors"):
            self.mode = "motors"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"
        else:
            raise RuntimeError("No valid drive function found on MBot")

    def send(self, vx):
        """Send forward/backward velocity (no rotation)."""
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, 0.0, 0.0)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, 0.0)
        elif self.mode == "motors":
            self.bot.motors(vx, vx)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0)

def main():
    robot = MBot()
    driver = Driver(robot)

    print(f"[INFO] Starting 1D Follow-Me Controller | Target={SETPOINT} m")

    try:
        while True:
            # ---- READ LIDAR ----
            ranges, thetas = [], []
            robot.readLidarScan(ranges, thetas)   # <--- LiDAR data comes from here

            # ---- COMPUTE DISTANCE ----
            dist = find_fwd_dist(ranges, thetas)
            print(f"Distance: {dist:.3f} m")

            if not np.isfinite(dist):
                driver.stop()
                time.sleep(0.1)
                continue

            # ---- CONTROL LOGIC ----
            error = dist - SETPOINT
            if abs(error) <= TOLERANCE:
                vx = 0.0
            else:
                vx = KP * error
                if vx > 0:
                    vx = clamp(vx, 0.0, MAX_FWD)
                else:
                    vx = clamp(vx, MAX_REV, 0.0)

            driver.send(vx)
            print(f"Command velocity: {vx:.3f} m/s")

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping robot.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()

