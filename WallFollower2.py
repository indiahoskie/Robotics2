#!/usr/bin/env python3
"""
Wall Follower (Python, MBot-Omni)
- Robust import/name shims for MBot + LiDAR/drive methods
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
        # some expose MBot inside .api
        if hasattr(module, "api") and hasattr(module.api, "MBot"):
            MBot = getattr(module.api, "MBot")
            break
    except Exception as e:
        _import_err = e

if MBot is None:
    raise SystemExit(
        "MBot SDK not found. Install the MBot Python package on the ROBOT:\n"
        "  python3 -m pip install mbot-bridge  (or mbot_bridge)\n"
        "or install your course’s SDK repo with: python3 -m pip install -e .\n\n"
        f"Last import error: {_import_err}"
    )

# ===================== CONFIG =====================
TRACK_LEFT = True          # True = follow left wall, False = right wall
USE_BANG_BANG = False      # True = Bang-Bang, False = P-control
SETPOINT = 0.50            # meters
DEAD_BAND = 0.04
KP = 0.9
BANG_V = 0.25
V_TAN = 0.22
V_MAX = 0.45
CLIP_MAX_RANGE = 8.0
LOOP_HZ = 15
# ==================================================

# -------- Small helpers --------
def unit2(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-6: return 0.0, 0.0
    return vx/n, vy/n

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def cross_product(a, b):
    ax, ay, az = a; bx, by, bz = b
    return [ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx]

# -------- API name shims --------
def read_lidar(bot) -> Tuple[list, list]:
    """
    Try common LiDAR method names; return (ranges, thetas).
    """
    for name in ("read_lidar", "readLidar", "read_lidar_scan", "readLidarScan", "get_lidar"):
        if hasattr(bot, name):
            out = getattr(bot, name)()
            # expect tuple (ranges, thetas)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            # some APIs return dict
            if isinstance(out, dict) and "ranges" in out and "thetas" in out:
                return out["ranges"], out["thetas"]
    raise RuntimeError("No LiDAR read method found (tried read_lidar/readLidar/read_lidar_scan/readLidarScan/get_lidar).")

def drive(bot, vx: float, vy: float, wz: float):
    """
    Try common drive command names.
    """
    for name in ("drive", "cmd_vel", "set_cmd_vel", "set_velocity"):
        if hasattr(bot, name):
            getattr(bot, name)(vx, vy, wz)
            return
    # fallback: many platforms use (vx, vy, wz) in 'drive'
    if hasattr(bot, "drive"):
        bot.drive(vx, vy, wz)
    else:
        raise RuntimeError("No drive method found (tried drive/cmd_vel/set_cmd_vel/set_velocity).")

# -------- Perception --------
def find_min_dist(ranges: List[float], thetas: List[float]) -> Tuple[float, float]:
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0
    rg = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)
    if CLIP_MAX_RANGE is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE)
    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0
    idx = np.nonzero(valid)[0][np.argmin(rg[valid])]
    return float(rg[idx]), float(th[idx])

# -------- Controller --------
def compute_wall_follower_command(
    ranges: List[float],
    thetas: List[float],
    setpoint_m: float,
    track_left: bool,
    use_bang_bang: bool = False,
    kp: float = 0.9,
    bang_v: float = 0.25,
    v_tan: float = 0.22,
    v_max: float = 0.45,
    dead_band: float = 0.04,
) -> Tuple[float, float, float]:

    d_min, th_min = find_min_dist(ranges, thetas)
    if not math.isfinite(d_min):
        return 0.0, 0.0, 0.0

    nx, ny = math.cos(th_min), math.sin(th_min)     # wall normal
    tx, ty, _ = cross_product([nx, ny, 0.0], [0.0, 0.0, 1.0])  # tangent = n x k
    if tx < 0.0:  # prefer forward-ish
        tx, ty = -tx, -ty

    tx, ty = unit2(tx, ty)
    nxu, nyu = unit2(nx, ny)

    error = setpoint_m - d_min   # + if too far, - if too close
    if not track_left:
        error = -error           # mirror for right-wall tracking

    if abs(error) <= dead_band:
        corr = 0.0
    else:
        corr = (bang_v if use_bang_bang else kp * error)

    vx_cmd = clamp(v_tan * tx + corr * nxu, -v_max, v_max)
    vy_cmd = clamp(v_tan * ty + corr * nyu, -v_max, v_max)
    wz_cmd = 0.0
    return vx_cmd, vy_cmd, wz_cmd

# -------- Main --------
def main():
    bot = MBot()
    print(f"[INFO] Wall follower: side={'LEFT' if TRACK_LEFT else 'RIGHT'}, "
          f"mode={'Bang-Bang' if USE_BANG_BANG else 'P'}, setpoint={SETPOINT:.2f} m")
    try:
        while True:
            ranges, thetas = read_lidar(bot)
            vx, vy, wz = compute_wall_follower_command(
                ranges, thetas,
                setpoint_m=SETPOINT,
                track_left=TRACK_LEFT,
                use_bang_bang=USE_BANG_BANG,
                kp=KP, bang_v=BANG_V, v_tan=V_TAN, v_max=V_MAX, dead_band=DEAD_BAND
            )
            drive(bot, vx, vy, wz)
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
