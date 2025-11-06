#!/usr/bin/env python3
"""
Drive forward and follow a wall using LiDAR (left or right).
- Auto-picks the nearer wall (within MAX_TRACK_DIST) unless locked
- Maintains a side distance (SETPOINT) using two-beam wall estimation
- If obstacle directly ahead -> stop or reverse
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------- CONFIGURATION ----------------
# Distances (meters)
SETPOINT        = 0.50   # desired distance to wall (side)
MAX_TRACK_DIST  = 2.50   # ignore walls farther than this when picking a side
FRONT_STOP_DIST = 0.60   # stop if something is this close ahead
FRONT_BACKUP_DIST = 0.35 # back up if something is this close ahead

# Speeds
FWD_SPEED   = 0.25       # forward speed (m/s) when cruising
REV_SPEED   = -0.18      # reverse speed (m/s)
WZ_MAX      = 0.8        # max yaw rate (rad/s)

# Control gains
KP_DIST   = 1.4          # side distance P-gain
K_ALPHA   = 0.9          # wall alignment gain (angle term)
BIAS_WZ   = 0.0          # small constant bias if your robot drifts

# LiDAR sampling
SECTOR_HALF_WIDTH_DEG = 20  # half width for sector averaging
# Two-beam angles (degrees) for wall estimation on each side
LEFT_THETA_DEG  = 70
LEFT_DELTA_DEG  = 30       # second ray at 70+30 = 100 deg
RIGHT_THETA_DEG = -70
RIGHT_DELTA_DEG = -30      # second ray at -70-30 = -100 deg

LOOP_HZ = 12
PRINT_EVERY = 6            # print every N loops
# ------------------------------------------------


def sector_mean(ranges, thetas, center_deg, half_width_deg):
    """Mean range in an angular sector (degrees). Returns inf if none."""
    if not ranges or not thetas:
        return float('inf')
    th = np.asarray(thetas)
    rg = np.asarray(ranges, dtype=float)
    w = math.radians(half_width_deg)
    c = math.radians(center_deg)
    mask = (rg > 0) & (th > c - w) & (th < c + w)
    if not np.any(mask):
        return float('inf')
    return float(np.mean(rg[mask]))


def interp_range_at_angle(ranges, thetas, target_rad):
    """
    Get an interpolated range at a desired angle (radians).
    Falls back to nearest ray if interpolation insufficient.
    Returns np.inf if no valid data.
    """
    if not ranges or not thetas:
        return float('inf')

    th = np.asarray(thetas)
    rg = np.asarray(ranges, dtype=float)
    valid = rg > 0
    if not np.any(valid):
        return float('inf')

    # Find nearest two valid indices around target
    idx = np.searchsorted(th, target_rad)
    candidates = []

    # Collect a few neighbors to try to interpolate
    for k in [idx-2, idx-1, idx, idx+1, idx+2]:
        if 0 <= k < len(th) and valid[k]:
            candidates.append((th[k], rg[k]))

    if not candidates:
        # fallback to nearest by absolute angle diff
        k = int(np.argmin(np.abs(th - target_rad)))
        val = rg[k]
        return float(val) if val > 0 else float('inf')

    # If we have at least two, do simple linear fit r(theta)
    if len(candidates) >= 2:
        thetas_c, ranges_c = zip(*sorted(candidates))
        thetas_c = np.array(thetas_c)
        ranges_c = np.array(ranges_c)
        # least squares fit: r = a*theta + b
        A = np.vstack([thetas_c, np.ones_like(thetas_c)]).T
        a, b = np.linalg.lstsq(A, ranges_c, rcond=None)[0]
        r_hat = a * target_rad + b
        return float(r_hat if r_hat > 0 else float('inf'))

    # Otherwise just return that single candidate
    return float(candidates[0][1])


def wall_estimate(ranges, thetas, side='left'):
    """
    Estimate wall angle (alpha) and perpendicular distance (D) using two beams.
    side: 'left' or 'right'
    Returns (D, alpha) where:
      - D is perpendicular distance from robot to wall (m)
      - alpha is wall orientation relative to robot x-axis (rad), 0 = parallel to x
    Based on standard two-point wall follower geometry (F1TENTH style).
    """
    if side == 'left':
        theta = math.radians(LEFT_THETA_DEG)
        delta = math.radians(LEFT_DELTA_DEG)
    else:
        theta = math.radians(RIGHT_THETA_DEG)
        delta = math.radians(RIGHT_DELTA_DEG)

    r1 = interp_range_at_angle(ranges, thetas, theta)        # at theta
    r2 = interp_range_at_angle(ranges, thetas, theta + delta) # at theta + delta

    if not np.isfinite(r1) or not np.isfinite(r2):
        return float('inf'), 0.0

    # alpha is the angle between the robot x-axis and the wall
    # Formula: alpha = atan2(r1*cos(Δ) - r2, r1*sin(Δ))
    alpha = math.atan2(r1 * math.cos(delta) - r2, r1 * math.sin(delta))

    # For the side distance: D = r2 * cos(alpha)
    D = r2 * math.cos(alpha)

    # If tracking the right wall, flip the sign of alpha so the controller behaves symmetrically
    if side == 'right':
        alpha = -alpha

    return float(D), float(alpha)


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
        if self.mode == "drive_vx_vy_wz":
            self.bot.drive(vx, 0.0, wz)
        elif self.mode == "drive_vx":
            self.bot.drive(vx)  # NOTE: yaw ignored if platform lacks yaw API
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
            # crude differential mapping
            left = vx - 0.25 * wz
            right = vx + 0.25 * wz
            self.bot.motors(left, right)

    def stop(self):
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0)


def main():
    bot = MBot()
    driver = Driver(bot)
    print(f"[INFO] Wall follower: setpoint={SETPOINT:.2f} m")

    # State for side selection
    side_lock = None   # 'left' or 'right' once we commit; set to None to allow auto-switch
    loop = 0

    try:
        while True:
            ranges, thetas = bot.read_lidar()

            # Front safety
            front_dist = sector_mean(ranges, thetas, center_deg=0, half_width_deg=SECTOR_HALF_WIDTH_DEG)

            # Choose side (if not locked): nearer wall within MAX_TRACK_DIST
            left_sector  = sector_mean(ranges, thetas, center_deg=90,  half_width_deg=SECTOR_HALF_WIDTH_DEG)
            right_sector = sector_mean(ranges, thetas, center_deg=-90, half_width_deg=SECTOR_HALF_WIDTH_DEG)

            if side_lock is None:
                candidates = []
                if np.isfinite(left_sector) and left_sector < MAX_TRACK_DIST:
                    candidates.append(('left', left_sector))
                if np.isfinite(right_sector) and right_sector < MAX_TRACK_DIST:
                    candidates.append(('right', right_sector))
                if candidates:
                    side_lock = min(candidates, key=lambda x: x[1])[0]

            # Default to left if nothing seen yet (you can change this)
            if side_lock is None:
                side_lock = 'left'

            # Estimate wall geometry on the chosen side
            D, alpha = wall_estimate(ranges, thetas, side=side_lock)

            # Compute control
            vx = FWD_SPEED
            wz = 0.0

            # Front collision handling
            if np.isfinite(front_dist) and front_dist <= FRONT_BACKUP_DIST:
                vx = REV_SPEED
                wz = 0.0
                action = "BACKUP (front too close)"
            elif np.isfinite(front_dist) and front_dist <= FRONT_STOP_DIST:
                vx = 0.0
                wz = 0.0
                action = "STOP (front obstacle)"
            else:
                # Side distance control only if we have a sane estimate
                if np.isfinite(D):
                    e = SETPOINT - D          # positive if too far from wall
                    # Combine distance and alignment (alpha) terms
                    wz_cmd = KP_DIST * e + K_ALPHA * alpha + BIAS_WZ
                    # Clamp
                    wz = max(-WZ_MAX, min(WZ_MAX, wz_cmd))
                    action = f"FOLLOW {side_lock.upper()} (e={e:+.2f}, alpha={alpha:+.2f})"
                else:
                    # No valid side reading → creep forward carefully
                    vx = 0.15
                    wz = 0.0
                    action = "SEARCH (no side data)"

            driver.send(vx, wz)

            # periodic log
            if (loop % PRINT_EVERY) == 0:
                ls = left_sector if np.isfinite(left_sector) else float('inf')
                rs = right_sector if np.isfinite(right_sector) else float('inf')
                fd = front_dist if np.isfinite(front_dist) else float('inf')
                print(f"[L:{ls:4.2f}  F:{fd:4.2f}  R:{rs:4.2f}]  side={side_lock}  D={D:4.2f}  α={alpha:+.2f}  → vx={vx:+.2f} wz={wz:+.2f}  | {action}")

            loop += 1
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


