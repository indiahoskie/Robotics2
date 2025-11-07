import time
import math
import numpy as np
from mbot_bridge.api import MBot


def find_min_dist(ranges, thetas):
    """Find the length and angle of the minimum VALID ray in the scan.
    Ignores zeros (invalid). Returns (min_dist, min_angle).
    """
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0

    rg = np.asarray(ranges, dtype=float)
    th = np.asarray(thetas, dtype=float)

    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0

    # index of the minimum among valid entries
    valid_idx = np.nonzero(valid)[0]
    k = valid_idx[np.argmin(rg[valid])]
    return float(rg[k]), float(th[k])


def cross_product(v1, v2):
    """Compute the Cross Product between two vectors (length 3)."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    return np.cross(a, b)


# --------- simple helpers ----------
def unit2(vx, vy):
    n = math.hypot(vx, vy)
    return (0.0, 0.0) if n < 1e-9 else (vx / n, vy / n)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def drive(robot, vx, vy, wz):
    """Call the appropriate drive method (course images vary)."""
    if hasattr(robot, "drive"):
        robot.drive(vx, vy, wz)
    elif hasattr(robot, "set_cmd_vel"):
        robot.set_cmd_vel(vx, vy, wz)
    elif hasattr(robot, "cmd_vel"):
        robot.cmd_vel(vx, vy, wz)
    else:
        raise RuntimeError("No drive command found on MBot (expected drive/set_cmd_vel/cmd_vel).")


robot = MBot()

# ---------------- TUNABLES ----------------
setpoint = 0.50       # desired distance to the wall (meters)
KP = 1.0              # lateral P gain
DEAD_BAND = 0.03      # meters; ignore tiny errors
V_TAN = 0.22          # m/s forward-along-wall speed
V_MAX = 0.45          # clamp for |vx|, |vy|
WZ_CMD = 0.0          # keep heading fixed; (you can add yaw control later)
CLIP_MAX_RANGE = 8.0  # optional sanity; ignore crazy long returns
LOOP_DT = 0.07        # ~14 Hz
# ------------------------------------------


try:
    print(f"[INFO] Following nearest wall with shortest-ray method. Setpoint={setpoint:.2f} m")
    while True:
        # Read the latest lidar scan.
        ranges, thetas = robot.read_lidar()

        # Optional: clip long weird returns for stability
        if CLIP_MAX_RANGE is not None:
            ranges = np.clip(ranges, 0.0, CLIP_MAX_RANGE).tolist()

        # 1) Find closest ray (distance & angle)
        d_min, th_min = find_min_dist(ranges, thetas)

        # If no valid reading, stop briefly and retry
        if not math.isfinite(d_min):
            drive(robot, 0.0, 0.0, 0.0)
            time.sleep(LOOP_DT)
            continue

        # 2) Build wall normal (toward obstacle) and tangent (parallel to wall)
        nx, ny = math.cos(th_min), math.sin(th_min)        # normal toward the wall
        tx, ty, _ = cross_product([nx, ny, 0.0], [0.0, 0.0, 1.0])  # t = n x k_hat = [ny, -nx, 0]

        # Prefer "forward-ish" tangent (positive x projection)
        if tx < 0.0:
            tx, ty = -tx, -ty

        # Normalize
        tx, ty = unit2(tx, ty)
        nxu, nyu = unit2(nx, ny)

        # 3) Lateral distance error: positive if we're too far from the wall
        error = setpoint - d_min

        # Deadband to avoid tiny oscillations
        if abs(error) <= DEAD_BAND:
            corr = 0.0
        else:
            corr = KP * error

        # 4) Compose velocity: go along-wall + small push toward/away from wall
        vx_cmd = V_TAN * tx + corr * nxu
        vy_cmd = V_TAN * ty + corr * nyu

        # clamp speeds
        vx_cmd = clamp(vx_cmd, -V_MAX, V_MAX)
        vy_cmd = clamp(vy_cmd, -V_MAX, V_MAX)

        # 5) Drive (no rotation for this assignment)
        drive(robot, vx_cmd, vy_cmd, WZ_CMD)

        # Optionally, sleep
        time.sleep(LOOP_DT)

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    try:
        robot.stop()
    except Exception:
        # fallback if stop() isn't available
        try:
            drive(robot, 0.0, 0.0, 0.0)
        except Exception:
            pass
    print("[INFO] Robot stopped.")


