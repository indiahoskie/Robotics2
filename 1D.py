#!/usr/bin/env python3
"""
Drive straight and steer away from obstacles using LiDAR.
- Drives forward by default
- If obstacle is on left -> moves right
- If obstacle is on right -> moves left
- If obstacle directly ahead -> stops or reverses
"""

import time
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------- CONFIGURATION ----------------
STOP_DIST = 0.5       # Stop if obstacle is closer than this (m)
BACKUP_DIST = 0.3     # Back up if object is closer than this (m)
FWD_SPEED = 0.25      # Forward speed (m/s)
REV_SPEED = -0.20     # Reverse speed (m/s)
TURN_SPEED = 0.4      # Angular velocity (rad/s) when avoiding
WINDOW = 8            # Number of rays near each side
LOOP_HZ = 10          # Control loop rate (Hz)
# ------------------------------------------------

def find_sector_distance(ranges, thetas, center=0, width_deg=30):
    """
    Average distance in a given angular sector.
    center = 0 (front), <0 (left), >0 (right)
    width_deg = half-width of the sector (degrees)
    """
    if not ranges or not thetas:
        return np.inf
    width = np.deg2rad(width_deg)
    thetas = np.array(thetas)
    ranges = np.array(ranges)
    valid = (ranges > 0) & (thetas > center - width) & (thetas < center + width)
    if not np.any(valid):
        return np.inf
    return float(np.mean(ranges[valid]))

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

    def send(self, vx, wz=0.0):
        """Send linear (vx) and angular (wz) velocity commands."""
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, 0.0, wz)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
            # crude approximation: wz changes wheel speeds
            left = vx - 0.2 * wz
            right = vx + 0.2 * wz
            self.bot.motors(left, right)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)
    print(f"[INFO] Starting: drives forward and steers away from obstacles (< {STOP_DIST} m)")

    try:
        while True:
            # ---- READ LIDAR ----
            ranges, thetas = bot.read_lidar()

            # ---- COMPUTE SECTOR DISTANCES ----
            left_dist = find_sector_distance(ranges, thetas, center=np.deg2rad(45))
            right_dist = find_sector_distance(ranges, thetas, center=np.deg2rad(-45))
            front_dist = find_sector_distance(ranges, thetas, center=0)

            print(f"L: {left_dist:.2f} | F: {front_dist:.2f} | R: {right_dist:.2f}")

            # ---- CONTROL LOGIC ----
            vx = FWD_SPEED
            wz = 0.0

            if not np.isfinite(front_dist):
                vx = FWD_SPEED * 0.5  # move cautiously if uncertain
            elif front_dist <= BACKUP_DIST:
                print("[ACTION] Too close! Backing up...")
                vx = REV_SPEED
            elif front_dist <= STOP_DIST:
                print("[ACTION] Obstacle ahead — stopping.")
                vx = 0.0
            else:
                # Side avoidance logic
                if left_dist < right_dist and left_dist < STOP_DIST:
                    print("[ACTION] Obstacle on LEFT — turning RIGHT.")
                    wz = -TURN_SPEED
                elif right_dist < left_dist and right_dist < STOP_DIST:
                    print("[ACTION] Obstacle on RIGHT — turning LEFT.")
                    wz = TURN_SPEED

            driver.send(vx, wz)
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
