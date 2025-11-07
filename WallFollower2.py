import time
import math
import numpy as np
from mbot_bridge.api import MBot


# ==========================
#  Function Definitions
# ==========================

def find_min_dist(ranges, thetas):
    """Find the shortest valid LiDAR ray (ignore zeros).
    Returns: (min_dist, min_angle)
    """
    min_dist, min_angle = None, None
    shortest = float('inf')

    for i in range(len(ranges)):
        if ranges[i] > 0 and ranges[i] < shortest:
            shortest = ranges[i]
            min_dist = ranges[i]
            min_angle = thetas[i]

    return min_dist, min_angle


def cross_product(v1, v2):
    """Compute the Cross Product between two 3D vectors."""
    res = np.zeros(3)
    res[0] = v1[1]*v2[2] - v1[2]*v2[1]
    res[1] = v1[2]*v2[0] - v1[0]*v2[2]
    res[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return res


# 🔹 Call cross_product right after defining it (as required)
# This is a basic example call to verify that the function works.
print("Cross product test (should be [0, 0, 1]):",
      cross_product([1, 0, 0], [0, 1, 0]))


# ==========================
#  Robot Control Section
# ==========================

robot = MBot()
setpoint = 0.50  # meters from wall
KP = 0.9         # proportional gain
DEAD_BAND = 0.03 # tolerance band
V_TAN = 0.22     # forward speed (m/s)
V_MAX = 0.45     # clamp speed
CLIP_MAX = 8.0   # ignore long returns

# small smoother
_prev_d = None
alpha = 0.3      # smoothing constant


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


try:
    print("[INFO] Starting wall follower (shortest ray + cross product).")

    while True:
        ranges, thetas = robot.read_lidar()

        # 1️⃣ Find the shortest valid distance & angle
        d_min, th_min = find_min_dist(ranges, thetas)
        if d_min is None or not math.isfinite(d_min):
            robot.drive(0, 0, 0)
            continue

        # Smooth the distance slightly to avoid twitching
        if _prev_d is None:
            d_used = d_min
        else:
            d_used = alpha * d_min + (1 - alpha) * _prev_d
        _prev_d = d_used

        # 2️⃣ Compute wall normal vector (toward wall)
        nx = math.cos(th_min)
        ny = math.sin(th_min)
        normal = [nx, ny, 0.0]

        # 3️⃣ Compute tangent using cross product: t = n × k̂
        tangent = cross_product(normal, [0.0, 0.0, 1.0])  # gives [ny, -nx, 0]
        tx, ty, _ = tangent

        # ensure tangent points generally forward
        if tx < 0:
            tx, ty = -tx, -ty

        # normalize both
        norm_n = math.hypot(nx, ny)
        norm_t = math.hypot(tx, ty)
        nx, ny = nx / norm_n, ny / norm_n
        tx, ty = tx / norm_t, ty / norm_t

        # 4️⃣ Compute distance error
        error = setpoint - d_used

        # 5️⃣ Lateral correction along normal
        corr = 0.0 if abs(error) <= DEAD_BAND else KP * error

        # 6️⃣ Final velocity: along wall tangent + normal correction
        vx = V_TAN * tx + corr * nx
        vy = V_TAN * ty + corr * ny

        # clamp speeds
        vx = clamp(vx, -V_MAX, V_MAX)
        vy = clamp(vy, -V_MAX, V_MAX)

        # 7️⃣ Drive the robot (no rotation)
        robot.drive(vx, vy, 0.0)

        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    robot.stop()
    print("[INFO] Robot stopped.")





