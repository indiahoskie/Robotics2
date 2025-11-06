#!/usr/bin/env python3
"""
MBot-Omni Wall Follower (Straight-Line, No Turning, PCA Fit)

- Tracks a single side wall ('left' or 'right') using LiDAR points in a side sector.
- Fits a line to those points via PCA to get a stable wall tangent and normal.
- Drives parallel to the wall with constant forward speed and lateral P-correction.
- Keeps heading fixed: wz = 0 (no rotation). Uses (vx, vy) only.
- Uses bot.read_lidar() -> (ranges, thetas)

This matches the assignment flow:
  1) Detect wall (choose a side sector, gather points)
  2) Find tangent via cross product / PCA (parallel to wall)
  3) Apply correction (P-control) toward/away from wall
  4) Convert to velocity and send
"""

import time
import math
import numpy as np
from typing import List, Tuple
from mbot_bridge.api import MBot

# ---------------- CONFIG ----------------
TRACK_SIDE = 'left'    # 'left' or 'right' — pick ONE side to follow and stay on it

# Sector selection (degrees, robot frame; +theta = CCW from +x)
LEFT_CENTER_DEG   = 90
RIGHT_CENTER_DEG  = -90
SECTOR_HALF_WIDTH = 25     # +/- half-width around the side center (deg)

# Range filtering (meters)
MIN_RANGE = 0.05
MAX_RANGE = 6.0

# Control targets
SETPOINT = 0.50           # desired perpendicular distance to wall (m)
DEAD_BAND = 0.03          # don't correct if |error| <= this

# Speeds (m/s)
V_TAN   = 0.25            # along-wall speed magnitude
VY_MAX  = 0.35            # max lateral correction magnitude
VX_MAX  = 0.45            # clamp safety
WZ_CMD  = 0.0             # lock heading (no turning)

# Controller gain (m -> m/s)
KP = 0.9

# Loop
LOOP_HZ = 15
# ----------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def side_sector_mask(thetas: np.ndarray, side: str) -> np.ndarray:
    """Boolean mask selecting rays in the left/right sector."""
    center_deg = LEFT_CENTER_DEG if side == 'left' else RIGHT_CENTER_DEG
    w = math.radians(SECTOR_HALF_WIDTH)
    c = math.radians(center_deg)
    return (thetas > c - w) & (thetas < c + w)


def collect_side_points(ranges: List[float], thetas: List[float], side: str) -> np.ndarray:
    """
    From LiDAR (ranges, thetas), select valid points in the chosen side sector
    and convert to Cartesian (x,y) in robot frame.
    Returns Nx2 array of points.
    """
    if not ranges or not thetas:
        return np.empty((0, 2))

    r = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)

    valid = (r > MIN_RANGE) & (r < MAX_RANGE)
    sector = side_sector_mask(th, side)
    mask = valid & sector
    if not np.any(mask):
        return np.empty((0, 2))

    r = r[mask]
    th = th[mask]

    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.stack([x, y], axis=1)


def pca_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a line to 2D points via PCA.
    Returns (t_hat, n_hat, D):
      t_hat: unit tangent vector along the wall (principal eigenvector)
      n_hat: unit normal pointing from robot toward the wall (sign chosen so D>0)
      D    : perpendicular distance from robot (origin) to the line (m, positive)
    The line passes through the point mean 'mu' with direction t_hat.
    """
    mu = points.mean(axis=0)  # line goes approximately through mu
    P = points - mu
    C = P.T @ P / max(len(points) - 1, 1)  # covariance-like

    # Eigen decomposition: principal direction = eigenvector with largest eigenvalue
    vals, vecs = np.linalg.eigh(C)
    idx_max = int(np.argmax(vals))
    t_hat = vecs[:, idx_max]   # 2-vector
    t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-9)

    # Normal is perpendicular to tangent: rotate t_hat by -90deg
    n_hat = np.array([t_hat[1], -t_hat[0]])
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-9)

    # Signed distance from origin to the line through mu with normal n_hat:
    # D_signed = n_hat · mu
    D_signed = float(n_hat @ mu)

    # Ensure n_hat points from robot toward the wall (D should be positive)
    if D_signed < 0:
        n_hat = -n_hat
        D_signed = -D_signed

    # For left-wall tracking, we prefer the wall to be on +y side; for right, on -y.
    # If that isn't the case, flip both t_hat and n_hat to maintain handedness.
    if TRACK_SIDE == 'left' and (mu[1] < 0):
        t_hat = -t_hat
        n_hat = -n_hat
        D_signed = float(-n_hat @ mu)  # recompute after flip to keep positive
    elif TRACK_SIDE == 'right' and (mu[1] > 0):
        t_hat = -t_hat
        n_hat = -n_hat
        D_signed = float(-n_hat @ mu)

    # Prefer forward-going tangent (positive x projection), so we don't drive backward
    if t_hat[0] < 0:
        t_hat = -t_hat  # flip along-wall direction only (normal stays as set above)

    return t_hat, n_hat, D_signed


def main():
    bot = MBot()
    print(f"[INFO] Wall follower (PCA) starting. Side={TRACK_SIDE}, setpoint={SETPOINT:.2f} m, no turning (wz=0)")

    try:
        while True:
            # Read LiDAR
            ranges, thetas = bot.read_lidar()

            # Gather side points
            pts = collect_side_points(ranges, thetas, TRACK_SIDE)

            if pts.shape[0] < 10:
                # Not enough points — creep forward cautiously, no lateral correction
                bot.drive(0.15, 0.0, WZ_CMD)
                time.sleep(1.0 / LOOP_HZ)
                continue

            # Fit wall line via PCA
            t_hat, n_hat, D = pca_line(pts)   # D is positive distance to wall

            # P-control on perpendicular distance
            error = SETPOINT - D  # + if too far; - if too close
            if abs(error) <= DEAD_BAND:
                corr = 0.0
            else:
                corr = KP * error

            # Build velocity in robot frame:
            #   along-wall:  V_TAN * t_hat
            #   correction:  corr * n_hat  (strafe toward wall if too far, away if too close)
            vx_cmd = V_TAN * t_hat[0] + corr * n_hat[0]
            vy_cmd = V_TAN * t_hat[1] + corr * n_hat[1]

            # Clamp
            vx_cmd = clamp(vx_cmd, -VX_MAX, VX_MAX)
            vy_cmd = clamp(vy_cmd, -VY_MAX, VY_MAX)

            # Send (no turn)
            bot.drive(vx_cmd, vy_cmd, WZ_CMD)

            # Loop rate
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

