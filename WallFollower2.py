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
    print("[INFO] Starting wall follow




