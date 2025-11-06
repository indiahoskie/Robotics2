#!/usr/bin/env python3
"""
Wall Follower (Bang-Bang or P-control) for MBot-Omni

Assignment alignment:
- P1.2.1: find_min_dist() returns (min_range, min_theta) ignoring invalid (zero) rays
- P1.2.2: cross_product(a, b) returns a x b for 3D vectors
- P1.2.3: Follows the nearest wall by:
    1) Finding closest ray (distance & angle)
    2) Using cross product to get a tangent vector (parallel to wall)
    3) Applying Bang-Bang or P-control correction toward/away from wall
    4) Converting that vector to a velocity command
Runs until Ctrl-C.
"""

import time
import math
from typing import List, Tuple

import numpy as np
from mbot_bridge.api import MBot

# ---------------- CONFIG ----------------
# Control mode
USE_BANG_BANG = False   # True = Bang-Bang, False = P-control

# Distance target
SETPOINT = 0.50         # desired distance (m) from the wall (closest obstacle)
DEAD_BAND = 0.04        # no correction if |error| <= DEAD_BAND

# P-control gain (only if USE_BANG_BANG == False)
KP = 0.9                # correction speed (m/s) per meter of error

# Bang-Bang lateral correction speed
BANG_V = 0.25           # magnitude of lateral correction (m/s)

# Forward motion along wall (tangent)
V_TAN = 0.22            # base forward (tangent) speed (m/s)

# Speed clamps
V_MAX = 0.45            # cap each component |vx|, |vy|
WZ_CMD = 0.0            # keep heading stable (no spin)

# LiDAR handling
CLIP_MAX_RANGE = 8.0    # clip absurdly large returns to stabilize min search
# ----------------------------------------


def find_min_dist(ranges: List[float], thetas: List[float]) -> Tuple[float, float]:
    """
    P1.2.1: Find the minimum VALID ray (range > 0).
    Returns (min_range, min_theta). If no valid rays, returns (inf, 0.0).
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

    # index within valid subset
    idx_valid_min = int(np.argmin(rg[valid]))
    # map to original index
    valid_idx = np.nonzero(valid)[0]
    k = valid_idx[idx_valid_min]

    return float(rg[k]), float(th[k])


def cross_product(a: List[float], b: List[float]) -> List[float]:
    """
    P1.2.2: 3D cross product: a x b.
    a, b are length-3 vectors [ax, ay, az], [bx, by, bz].
    Returns [cx, cy, cz].
    """
    ax, ay, az = a
    bx, by, bz = b
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return [cx, cy, cz]


def unit(vx: float, vy: float) -> Tuple[float, float]:
    """Normalize a 2D vector (vx, vy). If near zero, return (0, 0)."""
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return 0.0, 0.0
    return vx / n, vy / n


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def main():
    bot = MBot()
    print("[INFO] Wall follower starting. Ctrl-C to stop.")
    print(f"[INFO] Mode: {'Bang-Bang' if USE_BANG_BANG else 'P-control'}; setpoint={SETPOINT:.2f} m")

    try:
        while True:
            # 1) Read LiDAR (UPDATED: bot.read_lidar() returns (ranges, thetas))
            ranges, thetas = bot.read_lidar()

            # 2) Find closest wall point (distance & direction)
            d_min, th_min = find_min_dist(ranges, thetas)

            # Defensive: if no data, stop this cycle
            if not math.isfinite(d_min):
                bot.drive(0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            # Wall "normal" (pointing from robot toward wall)
            nx, ny = math.cos(th_min), math.sin(th_min)
            n_vec = [nx, ny, 0.0]

            # 3) Tangent direction along the wall via cross product with +z
            #    t = n x k_hat = [ny, -nx, 0]
            tx, ty, _ = cross_product(n_vec, [0.0, 0.0, 1.0])
            # Choose forward-facing tangent (positive x preference)
            if tx < 0.0:
                tx, ty = -tx, -ty
            tx, ty = unit(tx, ty)

            # 4) Side-distance error & correction along normal
            error = SETPOINT - d_min  # + if too far from wall; - if too close
            if abs(error) <= DEAD_BAND:
                corr = 0.0
            else:
                corr = (BANG_V if USE_BANG_BANG else KP * error)

            # Correction vector along ±normal (move toward wall if too far; away if too close)
            nxu, nyu = unit(nx, ny)
            vx_corr = corr * nxu
            vy_corr = corr * nyu

            # 5) Compose final velocity: tangent + correction
            vx_cmd = V_TAN * tx + vx_corr
            vy_cmd = V_TAN * ty + vy_corr

            # Clamp speeds
            vx_cmd = clamp(vx_cmd, -V_MAX, V_MAX)
            vy_cmd = clamp(vy_cmd, -V_MAX, V_MAX)

            # 6) Send velocity (keep heading stable)
            bot.drive(vx_cmd, vy_cmd, WZ_CMD)

            time.sleep(0.05)  # ~20 Hz

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

