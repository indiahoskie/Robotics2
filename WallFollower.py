#!/usr/bin/env python3
"""
Drive forward and FOLLOW A WALL (left or right) using LiDAR.
- Auto-picks the nearer wall (within MAX_TRACK_DIST) unless locked
- Maintains side distance (SETPOINT) via two-beam wall estimation
- Robust front safety using r*cos(theta) projection + debounce + hysteresis
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------- CONFIGURATION ----------------
# Distances (meters)
SETPOINT           = 0.50   # desired distance to wall on the tracked side
MAX_TRACK_DIST     = 2.50   # ignore walls farther than this when picking a side
FRONT_STOP_DIST    = 0.60   # stop if projected front clearance <= this
FRONT_BACKUP_DIST  = 0.35   # back up if projected front clearance <= this
RESUME_CLEAR_DIST  = 0.80   # hysteresis: need >= this to resume after a stop/backup

# Speeds
FWD_SPEED   = 0.25          # forward speed (m/s)
REV_SPEED   = -0.18         # reverse speed (m/s)
WZ_MAX      = 0.8           # max yaw rate (rad/s)

# Control gains
KP_DIST   = 1.4             # side distance P-gain
K_ALPHA   = 0.9             # alignment gain (wall angle term)
BIAS_WZ   = 0.0             # small constant bias if robot drifts

# LiDAR sampling
SECTOR_HALF_WIDTH_DEG = 20   # for side sector means
FRONT_HALF_WIDTH_DEG  = 10   # narrow front cone for clearance
CLIP_MAX_RANGE_M      = 8.0  # clip extreme/garbage long ranges

# Two-beam angles (degrees) for wall estimation on each side
LEFT_THETA_DEG   = 70
LEFT_DELTA_DEG   = 30        # second ray at 70+30 = 100 deg
RIGHT_THETA_DEG  = -70
RIGHT_DELTA_DEG  = -30       # second ray at -70-30 = -100 deg

# Debounce
FRONT_DEBOUNCE_N = 3         # need N consecutive hits before acting

# Looping / logging
LOOP_HZ     = 12
PRINT_EVERY = 6              # print every N loops
# ------------------------------------------------


def sector_mean(ranges, thetas, center_deg, half_width_deg):
    """Mean range in an angular sector (degrees). Returns inf if none."""
    if not ranges or not thetas:
        return float('inf')
    th = np.asarray(thetas)
    rg = np.asarray(ranges, dtype=float)
    # clip long junk to stabilize means
    if CLIP_MAX_RANGE_M is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE_M)
    w = math.radians(half_width_deg)
    c = math.radians(center_deg)
    mask = (rg > 0) & (th > c - w) & (th < c + w)
    if not np.any(mask):
        return float('inf')
    return float(np.mean(rg[mask]))


def interp_range_at_angle(ranges, thetas, target_rad):
    """
    Interpolate range at a desired angle (radians).
    Falls back to nearest valid ray if interpolation insufficient.
    Returns np.inf if no valid data.
    """
    if not ranges or not thetas:
        return float('inf')

    th = np.asarray(thetas)
    rg = np.asarray(ranges, dtype=float)
    if CLIP_MAX_RANGE_M is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE_M)

    valid = rg > 0
    if not np.any(valid):
        return float('inf')

    idx = np.searchsorted(th, target_rad)
    candidates = []
    for k in [idx-2, idx-1, idx, idx+1, idx+2]:
        if 0 <= k < len(th) and valid[k]:
            candidates.append((th[k], rg[k]))

    if not candidates:
        # fallback to nearest by abs angle diff
        k = int(np.argmin(np.abs(th - target_rad)))
        val = rg[k]
        return float(val) if val > 0 else float('inf')

    if len(candidates) >= 2:
        thetas_c, ranges_c = zip(*sorted(candidates))
        thetas_c = np.array(thetas_c)
        ranges_c = np.array(ranges_c)
        A = np.vstack([thetas_c, np.ones_like(thetas_c)]).T
        a, b = np.linalg.lstsq(A, ranges_c, rcond=None)[0]
        r_hat = a * target_rad + b
        return float(r_hat if r_hat > 0 else float('inf'))

    return float(candidates[0][1])


def wall_estimate(ranges, thetas, side='left'):
    """
    Estimate wall angle (alpha) and perpendicular distance (D) using two beams.
    side: 'left' or 'right'
    Returns (D, alpha) where:
      D     = perpendicular distance to wall (m)
      alpha = wall orientation wrt robot x-axis (rad); 0 ~ parallel to x
    """
    if side == 'left':
        theta = math.radians(LEFT_THETA_DEG)
        delta = math.radians(LEFT_DELTA_DEG)
    else:
        theta = math.radians(RIGHT_THETA_DEG)
        delta = math.radians(RIGHT_DELTA_DEG)

    r1 = interp_range_at_angle(ranges, thetas, theta)          # at theta
    r2 = interp_range_at_angle(ranges, thetas, theta + delta)  # at theta+delta

    if not np.isfinite(r1) or not np.isfinite(r2):
        return float('inf'), 0.0

    # Standard two-beam geometry
    alpha = math.atan2(r1 * math.cos(delta) - r2, r1 * math.sin(delta))
    D = r2 * math.cos(alpha)

    # Make controller symmetric: flip alpha for right wall
    if side == 'right':
        alpha = -alpha

    return float(D), float(alpha)


def front_clearance(ranges, thetas, half_width_deg=FRONT_HALF_WIDTH_DEG):
    """
    Robust 'front distance' using forward projection r*cos(theta) within a narrow cone.
    Returns the median of projected distances (largely immune to side-wall leakage).
    """
    if not ranges or not thetas:
        return float('inf')

    th = np.asarray(thetas)
    rg = np.asarray(ranges, dtype=float)
    if CLIP_MAX_RANGE_M is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE_M)

    w = math.radians(half_width_deg)
    mask = (np.abs(th) <= w) & (rg > 0.03)  # ignore zeros/tiny junk
    if not np.any(mask):
        return float('inf')

    proj = rg[mask] * np.cos(th[mask])      # forward component
    proj = proj[proj > 0.0]
    if proj.size == 0:
        return float('inf')

    return float(np.median(proj))


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
            self.bot.drive(vx)  # yaw ignored if platform doesn't support
        elif self.mode == "set_velocity":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "motors":
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

    # State for side selection & safety debouncing
    side_lock = None                 # 'left' or 'right'
    front_hits_stop = 0
    front_hits_backup = 0
    last_safety_hold = False         # True after stop/backup until clear >= RESUME_CLEAR_DIST

    loop = 0

    try:
        while True:
            ranges, thetas = bot.read_lidar()

            # --- FRONT SAFETY (projected) ---
            front_dist = front_clearance(ranges, thetas)

            # --- SIDE SELECTION (auto unless locked) ---
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
            if side_lock is None:
                side_lock = 'left'  # default if nothing seen yet

            # --- WALL ESTIMATION ---
            D, alpha = wall_estimate(ranges, thetas, side=side_lock)

            # --- CONTROL ---
            vx = FWD_SPEED
            wz = 0.0
            action = "FOLLOW"

            # Debounced + hysteretic front safety
            if np.isfinite(front_dist) and front_dist <= FRONT_BACKUP_DIST:
                front_hits_backup += 1
                front_hits_stop = max(front_hits_stop - 1, 0)
                if front_hits_backup >= FRONT_DEBOUNCE_N:
                    vx = REV_SPEED
                    wz = 0.0
                    last_safety_hold = True
                    action = f"BACKUP (front={front_dist:.2f} m)"
            elif np.isfinite(front_dist) and front_dist <= FRONT_STOP_DIST:
                front_hits_stop += 1
                front_hits_backup = max(front_hits_backup - 1, 0)
                if front_hits_stop >= FRONT_DEBOUNCE_N:
                    vx = 0.0
                    wz = 0.0
                    last_safety_hold = True
                    action = f"STOP (front={front_dist:.2f} m)"
            else:
                # clear front; decay counters
                front_hits_stop = max(front_hits_stop - 1, 0)
                front_hits_backup = max(front_hits_backup - 1, 0)

                # If previously stopped/backed up, require larger clearance to resume
                if last_safety_hold and (not np.isfinite(front_dist) or front_dist < RESUME_CLEAR_DIST):
                    vx = 0.0
                    wz = 0.0
                    action = f"HOLD (waiting ≥ {RESUME_CLEAR_DIST:.2f} m; front={front_dist:.2f} m)"
                else:
                    last_safety_hold = False
                    # Side distance + alignment controller (if estimate valid)
                    if np.isfinite(D):
                        e = SETPOINT - D  # positive if too far from wall
                        wz_cmd = KP_DIST * e + K_ALPHA * alpha + BIAS_WZ
                        wz = max(-WZ_MAX, min(WZ_MAX, wz_cmd))
                        action = f"FOLLOW {side_lock.upper()} (e={e:+.2f}, α={alpha:+.2f})"
                    else:
                        vx = 0.15
                        wz = 0.0
                        action = "SEARCH (no side data)"

            driver.send(vx, wz)

            # --- LOGGING ---
            if (loop % PRINT_EVERY) == 0:
                ls = left_sector if np.isfinite(left_sector) else float('inf')
                rs = right_sector if np.isfinite(right_sector) else float('inf')
                fd = front_dist if np.isfinite(front_dist) else float('inf')
                dstr = f"{D:4.2f}" if np.isfinite(D) else " inf"
                print(f"[L:{ls:4.2f}  F:{fd:4.2f}  R:{rs:4.2f}]  side={side_lock}  D={dstr}  α={alpha:+.2f}  → vx={vx:+.2f} wz={wz:+.2f}  | {action}")

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
