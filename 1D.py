#!/usr/bin/env python3
"""
Drive straight until an obstacle is detected using LiDAR.
Stops or backs up based on distance in front of the robot.
"""

import time
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------- CONFIGURATION ----------------
STOP_DIST = 0.5       # Stop if obstacle is closer than this (meters)
BACKUP_DIST = 0.3     # Back up if object is closer than this (meters)
FWD_SPEED = 0.25      # Forward speed (m/s)
REV_SPEED = -0.20     # Reverse speed (m/s)
WINDOW = 8            # Number of rays near the front to average
LOOP_HZ = 10          # Control loop rate (Hz)
# ------------------------------------------------

def find_fwd_dist(ranges, thetas, window=WINDOW):
    """Compute the average forward distance from LiDAR data."""
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

# ---- Motion adapter ----
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
        """Send a pure forward/backward velocity command (no rotation)."""
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
    bot = MBot()
    driver = Driver(bot)
    print(f"[INFO] Starting: driving straight until obstacle < {STOP_DIST} m")

    try:
        while True:
            # ---- READ LIDAR ----
            ranges, thetas = bot.read_lidar()   # ✅ your LiDAR call

            # ---- COMPUTE DISTANCE ----
            dist = find_fwd_dist(ranges, thetas)
            print(f"Front distance: {dist:.2f} m")

            # ---- CONTROL ----
            if not np.isfinite(dist):
                vx = FWD_SPEED * 0.5
            elif dist <= BACKUP_DIST:
                print("[ACTION] Too close! Reversing...")
                vx = REV_SPEED
            elif dist <= STOP_DIST:
                print("[ACTION] Obstacle ahead — stopping.")
                vx = 0.0
            else:
                vx = FWD_SPEED

            driver.send(vx)
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
