#!/usr/bin/env python3
"""
P1.1: Drive a 1m x 1m square three times WITH CORNER ROTATIONS.

- Drives each side forward (vx > 0, vy = 0)
- Rotates in place ~90° at each corner using wz (rad/s)
- Uses simple timing (no sensors), so tweak SIDE_TIME and ROT_SPEED as needed
"""

import time
import math
import inspect
from mbot_bridge.api import MBot

# -------------------- TUNABLES --------------------
SIDE_LENGTH_M   = 1.0     # target side length (m) - used just for reference
VX              = 0.25    # forward speed (m/s)
SIDE_TIME       = 4.0     # seconds to approximate SIDE_LENGTH_M at VX (tune on your floor)

ROT_SPEED       = math.radians(90)   # rad/s for rotation (e.g., 90°/s)
TURN_ANGLE      = math.radians(90)   # 90° per corner
PAUSE           = 0.2     # small pause between segments
LAPS            = 3

# Safety clamps
VX_MAX = 0.6
WZ_MAX = math.radians(180)
# --------------------------------------------------


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Driver:
    """
    Minimal adapter so this works whether MBot exposes:
    - drive(vx, vy, wz)
    - drive(vx, wz)
    - drive(vx)
    - set_velocity(vx, wz)
    - motors(left, right)
    """
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None
        if hasattr(bot, "drive"):
            sig = inspect.signature(bot.drive)
            n = len(sig.parameters)
            if n >= 3:
                self.mode = "drive_vx_vy_wz"
            elif n == 2:
                self.mode = "drive_vx_wz"
            else:
                self.mode = "drive_vx"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"
        elif hasattr(bot, "motors"):
            self.mode = "motors"
        else:
            raise RuntimeError("No supported drive method found")

    def send(self, vx=0.0, vy=0.0, wz=0.0):
        vx = clamp(vx, -VX_MAX, VX_MAX)
        wz = clamp(wz, -WZ_MAX, WZ_MAX)
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, vy, wz)
        elif self.mode == "drive_vx_wz":
            self.bot.drive(vx, wz)
        elif self.mode == "drive_vx":
            # no yaw channel; best effort forward only
            self.bot.drive(vx)
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
            # differential approx: wz skews wheel speeds
            left  = vx - 0.25 * wz
            right = vx + 0.25 * wz
            self.bot.motors(left, right)

    def stop(self):
        try:
            if hasattr(self.bot, "stop"):
                self.bot.stop()
            else:
                self.send(0.0, 0.0, 0.0)
        except Exception:
            pass


def drive_straight(driver: Driver, duration_s: float):
    driver.send(VX, 0.0, 0.0)
    time.sleep(duration_s)
    driver.send(0.0, 0.0, 0.0)


def rotate_in_place(driver: Driver, angle_rad: float, speed_rad_s: float):
    """
    Timed open-loop rotation.
    Positive angle -> rotate CCW (wz > 0)
    Negative angle -> rotate CW (wz < 0)
    """
    if speed_rad_s <= 1e-6:
        return
    wz = clamp(math.copysign(abs(speed_rad_s), angle_rad), -WZ_MAX, WZ_MAX)
    t = abs(angle_rad) / abs(wz)
    driver.send(0.0, 0.0, wz)
    time.sleep(t)
    driver.send(0.0, 0.0, 0.0)


def main():
    bot = MBot()
    driver = Driver(bot)
    print("[INFO] Driving a square with corner rotations...")
    print(f"[INFO] VX={VX:.2f} m/s, SIDE_TIME={SIDE_TIME:.2f}s, ROT_SPEED={math.degrees(ROT_SPEED):.1f} deg/s")

    try:
        for lap in range(LAPS):
            print(f"[INFO] Lap {lap+1}/{LAPS}")
            for edge in range(4):
                # 1) Straight segment
                drive_straight(driver, SIDE_TIME)
                time.sleep(PAUSE)

                # 2) Rotate ~90 degrees for the corner, except after the last corner of the last lap
                is_last_corner = (lap == LAPS - 1 and edge == 3)
                if not is_last_corner:
                    rotate_in_place(driver, TURN_ANGLE, ROT_SPEED)
                    time.sleep(PAUSE)

        driver.stop()
        print("[INFO] Done.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        driver.stop()


if __name__ == "__main__":
    main()

