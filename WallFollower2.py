import time
import math
import numpy as np
from mbot_bridge.api import MBot

# =================== TUNING ===================
SETPOINT = 0.50          # desired distance to the wall (m)
KP = 0.9                 # proportional gain (0.7–1.1 typical)
KD = 0.25                # derivative gain to damp oscillations (0.15–0.35)
DEAD_BAND = 0.04         # no correction inside this band (m)

VX_FWD = 0.20            # constant forward speed (m/s) -> straight line
VY_MAX = 0.28            # max strafe speed (m/s)
VY_SLEW = 0.07           # max change in vy per loop (m/s) (slew limiter)

CLIP_MAX = 8.0           # clip long returns (m)
EMA_ALPHA = 0.35         # smoothing for side distance (0..1, higher reacts faster)
LOOP_DT = 0.07           # ~14 Hz

# Sector windows (degrees from robot forward)
LEFT_SECTOR  = ( 25.0, 135.0)
RIGHT_SECTOR = (-135.0, -25.0)

FRONT_STOP_DIST = 0.28   # stop if something directly ahead is too close (m)
# ==============================================


def _wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi

def _sector_mask(thetas, deg_min, deg_max):
    th = np.asarray([_wrap_pi(t) for t in thetas], float)
    rmin, rmax = math.radians(deg_min), math.radians(deg_max)
    if rmin <= rmax:
        return (th >= rmin) & (th <= rmax)
    else:
        return (th >= rmin) | (th <= rmax)

def _front_too_close(ranges, thetas, dist):
    rg = np.asarray(ranges, float); th = np.asarray(thetas, float)
    mask = _sector_mask(th, -10.0, 10.0)
    vals = rg[(rg > 0.0) & mask]
    return vals.size > 0 and float(np.min(vals)) < dist


def find_min_dist(ranges, thetas):
    """(Assignment helper) Shortest VALID ray in full scan (ignores zeros)."""
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0
    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    if CLIP_MAX is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX)
    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0
    idx = np.nonzero(valid)[0][np.argmin(rg[valid])]
    return float(rg[idx]), float(th[idx])

def cross_product(v1, v2):
    """3D cross product (for your rubric)."""
    a = np.asarray(v1, float); b = np.asarray(v2, float)
    return np.cross(a, b)

def _side_percentile_dist(ranges, thetas, side="left", pct=20.0):
    """
    Stable side distance: use percentile (e.g., 20th) within side sector.
    Returns (d_side, theta_of_shortest, ok)
    """
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0, False

    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    if CLIP_MAX is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX)

    mask = _sector_mask(th, *(LEFT_SECTOR if side == "left" else RIGHT_SECTOR))
    valid = (rg > 0.0) & mask
    if not np.any(valid):
        return float('inf'), 0.0, False

    vals = rg[valid]
    d_side = float(np.percentile(vals, pct))  # more stable than simple min

    # also return angle of the actual min in sector (in case you need it)
    idx_valid = np.nonzero(valid)[0]
    kmin = idx_valid[np.argmin(rg[valid])]
    return d_side, float(th[kmin]), True


# ====== Controller state ======
robot = MBot()
side_lock = None        # "left" or "right" (locked at startup)
_prev_d_ema = None
_prev_err = 0.0
_prev_vy = 0.0

print("[INFO] Starting wall follower (stable PD, side-locked).")

try:
    while True:
        ranges, thetas = robot.read_lidar()

        # 0) Safety: obstacle straight ahead?
        if _front_too_close(ranges, thetas, FRONT_STOP_DIST):
            robot.drive(0.0, 0.0, 0.0)
            time.sleep(LOOP_DT)
            continue

        # 1) Choose & lock side once (based on which side is closer initially)
        if side_lock is None:
            dL, _, okL = _side_percentile_dist(ranges, thetas, "left")
            dR, _, okR = _side_percentile_dist(ranges, thetas, "right")
            if okL and okR:
                side_lock = "left" if dL < dR else "right"
            elif okL:
                side_lock = "left"
            elif okR:
                side_lock = "right"
            else:
                # no good data yet: just roll forward
                robot.drive(VX_FWD, 0.0, 0.0)
                time.sleep(LOOP_DT)
                continue
            print(f"[INFO] Locked to {side_lock.upper()} wall.")

        # 2) Measure stable side distance (percentile inside locked side)
        d_side, th_min, ok = _side_percentile_dist(ranges, thetas, side_lock)
        if not ok or not math.isfinite(d_side):
            # lost wall; coast forward, no strafe
            vy_cmd = 0.0
            robot.drive(VX_FWD, vy_cmd, 0.0)
            time.sleep(LOOP_DT)
            continue

        # 3) Smooth distance & PD error
        if _prev_d_ema is None:
            d_ema = d_side
        else:
            d_ema = EMA_ALPHA * d_side + (1.0 - EMA_ALPHA) * _prev_d_ema
        _prev_d_ema = d_ema

        err = SETPOINT - d_ema                  # + if too far from wall
        derr = (err - _prev_err) / max(LOOP_DT, 1e-3)
        _prev_err = err

        # 4) Deadband + PD control mapped to fixed side
        if abs(err) <= DEAD_BAND:
            vy = 0.0
        else:
            side_sign = +1.0 if side_lock == "left" else -1.0
            vy = side_sign * (KP * err + KD * derr)

        # 5) Limit & slew vy (prevents back-and-forth)
        vy = max(-VY_MAX, min(VY_MAX, vy))
        # slew-rate limit:
        dv_max = VY_SLEW
        vy_cmd = _prev_vy + max(-dv_max, min(dv_max, vy - _prev_vy))
        _prev_vy = vy_cmd

        # 6) Drive straight with lateral correction; no rotation
        robot.drive(VX_FWD, vy_cmd, 0.0)

        time.sleep(LOOP_DT)

except KeyboardInterrupt:
    pass
finally:
    try:
        robot.stop()
    except Exception:
        try:
            robot.drive(0.0, 0.0, 0.0)
        except Exception:
            pass
    print("[INFO] Robot stopped.")



