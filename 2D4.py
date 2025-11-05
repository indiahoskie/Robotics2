#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Straight-Line Motion (for MBot differential drive)
- Robot turns in place to face nearest obstacle (using wheel speed difference)
- Then drives straight toward it (equal wheel speeds)
- Uses only bot.motors(left, right)
"""

import time
import math
import numpy as np
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
V_FWD_MAX      = 0.35      # forward top speed (m/s)
V_TURN_MAX     = 0.25      # max turn differential speed (m/s)
K_FWD          = 1.2       # proportional gain for forward motion
K_TURN         = 2.5       # proportional gain for rotation
TH_ALIGN       = 0.10      # radians; consider aligned when |theta| < TH_ALIGN
NEAR_STOP      = 0.05      # meters; stop when too close
LOOP_HZ        = 30
PRINT_DEBUG    = True
# ====================================================================

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def find_min_nonzero(ranges, thetas):
    """Return distance and angle of nearest nonzero LIDAR point."""
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0.0
    if not np.any(mask):
        return (float("inf"), 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    idx = int(np.argmin(r_valid))
    return float(r_valid[idx]), float(t_valid[idx])

def main():
    bot = MBot()
    print("[INFO] Follow-Me Straight-Line started (using motors).")

    try:
        dt = 1.0 / LOOP_HZ
        while True:
            # --- Read LiDAR ---
            ranges, thetas = bot.read_lidar()
            d, th = find_min_nonzero(ranges, thetas)

            if not np.isfinite(d):
                bot.motors(0.0, 0.0)
                time.sleep(dt)
                continue

            # --- Rotation control (turn to face target) ---
            if abs(th) > TH_ALIGN:
                # turn in place proportional to angle
                turn = clamp(K_TURN * th, -V_TURN_MAX, V_TURN_MAX)
                left = -turn
                right = turn
            else:
                # drive straight toward the object
                if d < NEAR_STOP:
                    left = 0.0
                    right = 0.0
                else:
                    fwd = clamp(K_FWD * d, 0.0, V_FWD_MAX)
                    left = fwd
                    right = fwd

            # --- Send to motors ---
            bot.motors(left, right)

            if PRINT_DEBUG:
                print(f"[CMD] left={left:+.3f}  right={right:+.3f}  | d={d:.3f} m, th={th:+.3f} rad")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        bot.motors(0.0, 0.0)
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()


