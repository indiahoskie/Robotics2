#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Straight-Line for Differential Drive
- Robot turns in place to face the nearest obstacle (using wz)
- Then drives straight toward it (vx)
- No lateral motion, no circling
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
VX_MAX        = 0.35      # forward top speed (m/s)
VX_MIN        = 0.05
KX            = 1.2       # distance P gain (for forward speed)
WZ_MAX        = 1.0       # max turn speed (rad/s)
KW            = 3.0       # angular P gain (for rotation)
TH_ALIGN      = 0.10      # radians; consider aligned when |theta| < TH_ALIGN
NEAR_STOP     = 0.05      # stop when very close (m)
LOOP_HZ       = 30
PRINT_DEBUG   = True
# ====================================================================

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def find_min_nonzero(ranges, thetas):
    """Find nearest obstacle distance and its angle."""
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0
    if not np.any(mask):
        return (float("inf"), 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    i = np.argmin(r_valid)
    return float(r_valid[i]), float(t_valid[i])

class Driver:
    """Handles drive commands."""
    def __init__(self, bot: MBot):
        self.bot = bot
        if hasattr(bot, "drive"):
            self.mode = "drive"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"
        elif hasattr(bot, "motors"):
            self.mode = "motors"
        else:
            raise RuntimeError("No valid drive function found")

    def send(self, vx, wz):
        if self.mode == "drive":
            self.bot.drive(vx, wz)
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
            # convert to left/right wheel speeds
            L = vx - 0.5 * wz
            R = vx + 0.5 * wz
            self.bot.motors(L, R)

    def stop(self):
        try:
            self.bot.stop()
        except Exception:
            self.send(0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)

    print("[INFO] Follow-Me Straight-Line started (no circling).")

    try:
        dt = 1.0 / LOOP_HZ
        while True:
            ranges, thetas = bot.read_lidar()
            d, th = find_min_nonzero(ranges, thetas)

            if not np.isfinite(d):
                driver.stop()
                time.sleep(dt)
                continue

            # --- ROTATION CONTROL (align first) ---
            if abs(th) > TH_ALIGN:
                vx = 0.0
                wz = clamp(KW * th, -WZ_MAX, WZ_MAX)
            else:
                # --- FORWARD CONTROL (drive straight) ---
                if d < NEAR_STOP:
                    vx = 0.0
                else:
                    vx = clamp(KX * d, VX_MIN, VX_MAX)
                wz = 0.0  # no rotation while driving straight

            driver.send(vx, wz)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx:+.3f} m/s, wz={wz:+.3f} rad/s | d={d:.3f} m, th={th:+.3f} rad")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()


