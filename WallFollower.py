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
USE_BANG_BANG = False   # set True for Bang-Bang, False for P-control

# Distance target
SETPOINT = 0.50         # desired distance (m) from the wall (closest obstacle)
DEAD_BAND = 0.04        # no correction if |error| <= DEAD_BAND

# P-control gain (only if USE_BANG_BANG == False)
KP = 0.9                # correction speed (m/s) per meter of error

# Bang-Bang lateral correction speed (only if USE_BANG_BANG == True)
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

    # Clip big values (optional but helps with flaky scans)
    if CLIP_MAX_RANGE is not None:
        rg = np.clip(rg, 0.0, CLIP_MAX_RANGE)

    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0

    idx = int(np.argmin(rg[valid]))
    # map back to original indices
    valid_idx = np.nonzero(valid)[0]
    k = valid_idx[idx]

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
    robot = MBot()
    print("[INFO] Wall follower starting. Ctrl-C to stop.")
    print(f"[INFO] Mode: {'Bang-Bang' if USE_BANG_BANG else 'P-control'}; setpoint={SETPOINT:.2f} m")

    try:
        while True:
            # 1) Read LiDAR (UPDATED to use bot.read_lidar API)
            ranges, thetas = robot.read_lidar()

            # 2) Find closest wall point (distance & direction)
            d_min, th_min = find_min_dist(ranges, thetas)

            # Defensive: if no data, stop this cycle
            if not math.isfinite(d_min):
                robot.drive(0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            # Wall "normal" (pointing from robot toward wall)
            nx, ny = math.cos(th_min), math.sin(th_min)
            n_vec = [nx, ny, 0.0]

            # 3) Tangent direction along the wall (parallel to wall)
            #    Use cross product as requested: t = n x k_hat
            #    where k_hat = [0, 0, 1] is +z.
            tx, ty, _ = cross_product(n_vec, [0.0, 0.0, 1.0])  # = [ny, -nx, 0]
            # Pick the tangent pointing "forward" in x if possible (so we don't go backward)
            if tx < 0.0:
                tx, ty = -tx, -ty

            # Normalize tangent
            tx, ty = unit(tx, ty)

            # 4) Side-distance error & correction toward/away from wall (along normal)
            error = SETPOINT - d_min  # + if too far from wall; - if too close

            if abs(error) <= DEAD_BAND:
                corr = 0.0
            else:
                if USE_BANG_BANG:
                    corr = BANG_V if error > 0.0 else -BANG_V
                else:
                    corr = KP * error  # m -> m/s

            # Correction direction: move along +n if too far (error>0), along -n if too close
            # So lateral correction vector = corr * n_hat
            nxu, nyu = unit(nx, ny)
            vx_corr = corr * nxu
            vy_corr = corr * nyu

            # 5) Compose final velocity: tangent motion + correction
            vx_cmd = V_TAN * tx + vx_corr
            vy_cmd = V_TAN * ty + vy_corr

            # Clamp speeds
            vx_cmd = clamp(vx_cmd, -V_MAX, V_MAX)
            vy_cmd = clamp(vy_cmd, -V_MAX, V_MAX)

            # 6) Send velocity (keep heading stable)
            robot.drive(vx_cmd, vy_cmd, WZ_CMD)

            # Optional: small sleep for loop rate (~20 Hz)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C received. Stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        try:
            robot.drive(0.0, 0.0, 0.0)
        except Exception:
            pass
        print("[INFO] Robot stopped safely.")


if __name__ == "__main__":
    main()
