#!/usr/bin/env python3
"""
MBot Wall Follower (stable, no oscillation)
- Tracks ONLY the chosen side sector (left or right), not the global min
- Smooths side distance to suppress jitter
- Drives forward at constant V_TAN, strafes laterally to maintain setpoint
- wz = 0 (no turning)
"""

import time, math
from typing import List, Tuple
import numpy as np

# -------- Robust import for MBot --------
MBot = None
_import_err = None
for modpath in ("mbot_bridge.api", "mbot.api", "mbot_bridge", "mbot"):
    try:
        module = __import__(modpath, fromlist=["MBot"])
        if hasattr(module, "MBot"):
            MBot = getattr(module, "MBot")
            break
        if hasattr(module, "api") and hasattr(module.api, "MBot"):
            MBot = getattr(module.api, "MBot")
            break
    except Exception as e:
        _import_err = e
if MBot is None:
    raise SystemExit(
        "MBot SDK not found. Install on the ROBOT:\n"
        "  python3 -m pip install mbot-bridge  (or mbot_bridge)\n"
        "or install your course SDK repo: python3 -m pip install -e .\n\n"
        f"Last import error: {_import_err}"
    )

# ===================== CONFIG =====================
TRACK_LEFT = True          # True = follow left wall, False = right wall
USE_BANG_BANG = False      # keep P-control unless you must switch
SETPOINT = 0.50            # meters to wall (side distance)
DEAD_BAND = 0.03           # meters: no correction inside this band
KP = 1.0                   # P gain for lateral correction
BANG_V = 0.25              # bang-bang lateral speed (if enabled)

V_TAN = 0.20               # constant forward speed
VY_MAX = 0.35              # clamp for |vy|
VX_MAX = 0.35
WZ_CMD = 0.0               # keep heading fixed

CLIP_MAX_RANGE = 8.0       # ignore crazy long returns
FRONT_STOP_DIST = 0.30     # stop if something directly ahead is too close

# Side sector to look for the wall (degrees from forward, CCW positive)
# Left wall: +60..+120 deg ; Right wall: -120..-60 deg
SIDE_MIN_DEG = 60.0
SIDE_MAX_DEG = 120.0

# Smoothing
EMA_ALPHA = 0.3            # 0..1, higher = snappier
# Loop rate
LOOP_HZ = 15
# ==================================================

# -------- API name shims --------
def read_lidar(bot) -> Tuple[list, list]:
    for name in ("read_lidar", "readLidar", "read_lidar_scan", "readLidarScan", "get_lidar"):
        if hasattr(bot, name):
            out = getattr(bot, name)()
            if isinstance(out, tuple) and len(out) == 2:
                return out
            if isinstance(out, dict) and "ranges" in out and "thetas" in out:
                return out["ranges"], out["thetas"]
    raise RuntimeError("No LiDAR read method found.")

def drive(bot, vx: float, vy: float, wz: float):
    for name in ("drive", "cmd_vel", "set_cmd_vel", "set_velocity"):
        if hasattr(bot, name):
            getattr(bot, name)(vx, vy, wz)
            return
    if hasattr(bot, "drive"):
        bot.drive(vx, vy, wz)
    else:
        raise RuntimeError("No drive method found.")

# -------- Helpers --------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ang_norm(a: float) -> float:
    # wrap angle to [-pi, pi]
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a

def sector_mask(thetas: np.ndarray, deg_min: float, deg_max: float) -> np.ndarray:
    """Return boolean mask for angles within [deg_min, deg_max] in *degrees*,
       converting to radians and handling wrap correctly."""
    rmin = math.radians(deg_min)
    rmax = math.radians(deg_max)
    th = np.array([ang_norm(t) for t in thetas], dtype=float)
    # handle no wrap (e.g., +60..+120) vs wrap (e.g., -120..-60 also fine here)
    if rmin <= rmax:
        return (th >= rmin) & (th <= rmax)
    else:
        # wrapped range (not used in our chosen numbers, but safe)
        return (th >= rmin) | (th <= rmax)

def robust_side_distance(
    ranges: List[float],
    thetas: List[float],
    track_left: bool,
) -> Tuple[float, bool]:
    """Return (side_distance_m, valid). Uses only side sector and robust statistic."""
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float("inf"), False
    rg = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)

    if CLIP_MAX_RANGE is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE)

    # Pick sector based on side
    if track_left:
        m = sector_mask(th, SIDE_MIN_DEG, SIDE_MAX_DEG)   # +60..+120°
    else:
        # mirror to right: -120..-60 deg
        m = sector_mask(th, -SIDE_MAX_DEG, -SIDE_MIN_DEG)

    # valid = nonzero range & in sector
    valid = (rg > 0.0) & m
    if not np.any(valid):
        return float("inf"), False

    sector_vals = rg[valid]

    # Robust estimate: use median, not min
    d_side = float(np.median(sector_vals))
    return d_side, True

def front_too_close(ranges: List[float], thetas: List[float], stop_dist: float) -> bool:
    if not ranges or not thetas:
        return False
    rg = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)
    # consider a narrow front window ±10°
    mask = sector_mask(th, -10.0, 10.0)
    vals = rg[(rg > 0.0) & mask]
    if vals.size == 0:
        return False
    return np.min(vals) < stop_dist

# -------- Controller (side-distance only; no global-min chasing) --------
_prev_d_ema = None

def compute_vx_vy(
    d_side: float,
    valid: bool,
    setpoint_m: float,
    track_left: bool,
    kp: float,
    dead_band: float,
    use_bang_bang: bool,
    bang_v: float,
) -> Tuple[float, float]:
    global _prev_d_ema

    # Smooth the distance
    if not valid or not math.isfinite(d_side):
        # no sight of wall → no lateral correction
        return V_TAN, 0.0

    if _prev_d_ema is None:
        d_ema = d_side
    else:
        d_ema = EMA_ALPHA * d_side + (1.0 - EMA_ALPHA) * _prev_d_ema
    _prev_d_ema = d_ema

    # Positive error means "too far from the wall"
    error = setpoint_m - d_ema

    # Desired vy sign: +vy is left strafe in MBot frame.
    # If tracking LEFT: error>0 (too far) → +vy (move left toward wall)
    # If tracking RIGHT: mirror sign.
    if not track_left:
        error = -error

    if abs(error) <= dead_band:
        vy = 0.0
    else:
        vy = bang_v * (1 if error > 0 else -1) if use_bang_bang else kp * error

    vy = clamp(vy, -VY_MAX, VY_MAX)
    vx = clamp(V_TAN, -VX_MAX, VX_MAX)  # keep constant forward motion
    return vx, vy

# -------- Main --------
def main():
    bot = MBot()
    side = "LEFT" if TRACK_LEFT else "RIGHT"
    mode = "Bang-Bang" if USE_BANG_BANG else "P"
    print(f"[INFO] Wall follower: side={side}, mode={mode}, setpoint={SETPOINT:.2f} m")

    try:
        while True:
            ranges, thetas = read_lidar(bot)

            # Safety: stop if something too close in front
            if front_too_close(ranges, thetas, FRONT_STOP_DIST):
                drive(bot, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            d_side, ok = robust_side_distance(ranges, thetas, TRACK_LEFT)
            vx, vy = compute_vx_vy(
                d_side, ok, SETPOINT, TRACK_LEFT, KP, DEAD_BAND, USE_BANG_BANG, BANG_V
            )
            drive(bot, vx, vy, WZ_CMD)
            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C received. Stopping.")
    finally:
        try:
            drive(bot, 0.0, 0.0, 0.0)
        except Exception:
            pass
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()

