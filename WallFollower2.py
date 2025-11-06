#!/usr/bin/env python3
"""
Wall Follower (Python, MBot-Omni)
- Implements vector_add, cross_product, find_min_dist, compute_wall_follower_command
- Uses bot.read_lidar() -> (ranges, thetas)
- Drives straight along the wall (wz=0), strafes (vy) to maintain setpoint
- Controller selectable: Bang-Bang or P-control
"""

import time
import math
from typing import List, Tuple

import numpy as np
from mbot_bridge.api import MBot

# ===================== CONFIG =====================
# Choose side to hug visually (affects lateral sign mapping in compute_wall_follower_command)
TRACK_LEFT = True         # True = follow left wall, False = follow right wall

# Control mode
USE_BANG_BANG = False     # True for Bang-Bang, False for P-control

# Targets / gains
SETPOINT = 0.50           # desired distance (m) to wall
DEAD_BAND = 0.04          # no correction inside this band
KP = 0.9                  # P gain (only used if USE_BANG_BANG = False)
BANG_V = 0.25             # lateral correction speed for Bang-Bang (m/s)

# Motion
V_TAN = 0.22              # forward (tangent) speed (m/s)
V_MAX = 0.45              # clamp for |vx|, |vy|
WZ_CMD = 0.0              # keep heading fixed (no turning)

# LiDAR handling
CLIP_MAX_RANGE = 8.0      # clip long returns for stability

# Loop rate
LOOP_HZ = 15
# ==================================================


# ---------- Utility functions ----------
def vector_add(a: List[float], b: List[float]) -> List[float]:
    """3D vector add: a + b."""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def cross_product(a: List[float], b: List[float]) -> List[float]:
    """3D cross product: a x b."""
    ax, ay, az = a
    bx, by, bz = b
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def unit2(vx: float, vy: float) -> Tuple[float, float]:
    """Normalize a 2D vector (vx,vy)."""
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return 0.0, 0.0
    return vx / n, vy / n


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------- Sensing helpers ----------
def find_min_dist(ranges: List[float], thetas: List[float]) -> Tuple[float, float]:
    """
    Find the minimum VALID ray (range > 0). Returns (min_range, min_theta).
    """
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0

    rg = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)

    if CLIP_MAX_RANGE is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE)

    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0

    idx_valid_min = int(np.argmin(rg[valid]))
    valid_idx = np.nonzero(valid)[0]
    k = valid_idx[idx_valid_min]
    return float(rg[k]), float(th[k])


# ---------- Core controller ----------
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
    """
    Build (vx, vy, wz) to follow the nearest wall:
      1) Find closest ray (distance d_min, angle th_min)
      2) Wall normal n = [cos(th), sin(th), 0]
      3) Tangent t = n x k_hat (k_hat=[0,0,1]) -> parallel to wall
      4) Apply correction along n to hold setpoint
      5) vx, vy = v_tan * t + corr * n   ; wz = 0
    """
    d_min, th_min = find_min_dist(ranges, thetas)
    if not math.isfinite(d_min):
        return 0.0, 0.0, 0.0

    # Normal toward wall
    nx = math.cos(th_min)
    ny = math.sin(th_min)
    n_vec = [nx, ny, 0.0]

    # Tangent via cross product with +z: t = n x k̂ = [ny, -nx, 0]
    tx, ty, _ = cross_product(n_vec, [0.0, 0.0, 1.0])

    # Prefer forward tangent (positive x projection)
    if tx < 0.0:
        tx, ty = -tx, -ty

    # Normalize 2D directions
    tx, ty = unit2(tx, ty)
    nxu, nyu = unit2(nx, ny)

    # Distance error (using closest ray as "distance to wall")
    error = setpoint_m - d_min  # + if too far; - if too close

    # Map lateral sign to chosen side:
    # For left-wall tracking: if too far (error>0) we need +vy (left strafe)
    # For right-wall tracking: mirror the sign
    if not track_left:
        error = -error

    # Controller
    if abs(error) <= dead_band:
        corr = 0.0
    else:
        corr = (bang_v if use_bang_bang else kp * error)

    # Compose
    vx_cmd = v_tan * tx + corr * nxu
    vy_cmd = v_tan * ty + corr * nyu
    vx_cmd = clamp(vx_cmd, -v_max, v_max)
    vy_cmd = clamp(vy_cmd, -v_max, v_max)
    wz_cmd = 0.0  # keep heading straight

    return vx_cmd, vy_cmd, wz_cmd


# ---------- Main loop ----------
def main():
    bot = MBot()
    print(f"[INFO] Wall follower: side={'LEFT' if TRACK_LEFT else 'RIGHT'}, "
          f"mode={'Bang-Bang' if USE_BANG_BANG else 'P'}, setpoint={SETPOINT:.2f} m")

    try:
        while True:
            # READ LIDAR (Python API): returns (ranges, thetas)
            ranges, thetas = bot.read_lidar()

            vx, vy, wz = compute_wall_follower_command(
                ranges, thetas,
                setpoint_m=SETPOINT,
                track_left=TRACK_LEFT,
                use_bang_bang=USE_BANG_BANG,
                kp=KP,
                bang_v=BANG_V,
                v_tan=V_TAN,
                v_max=V_MAX,
                dead_band=DEAD_BAND,
            )

            bot.drive(vx, vy, wz)
            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C received. Stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        try:
            bot.drive(0.0, 0.0, 0.0)
        except Exception:
            pass
        print("[INFO] Robot stopped safely.")


if __name__ == "__main__":
    main()

