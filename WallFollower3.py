#!/usr/bin/env python3
import time, math, numpy as np
from mbot_bridge.api import MBot

# ================== REQUIRED FUNCS ==================

def find_min_dist(ranges, thetas):
    """Return (min_dist, min_angle) for the shortest VALID (range>0) ray."""
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0
    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    rg = np.clip(rg, 0.0, 8.0)               # sanity clip
    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0
    idx = np.nonzero(valid)[0][np.argmin(rg[valid])]
    return float(rg[idx]), float(th[idx])


def cross_product(v1, v2):
    """3D cross product (manual)."""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return [y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2]


# ---- required: call cross_product after definition (simple test) ----
_ = cross_product([1, 0, 0], [0, 1, 0])  # -> [0,0,1]

# ================== HELPERS ==================

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def unit2(x, y):
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < 1e-9 else (x/n, y/n)

def wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi

def front_min_dist(ranges, thetas, fov_deg=20.0):
    """Min distance in a +-fov/2 front sector; inf if none."""
    if not ranges or not thetas:
        return float('inf')
    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    th = (th + math.pi) % (2*math.pi) - math.pi
    half = math.radians(fov_deg/2.0)
    mask = (np.abs(th) <= half) & (rg > 0.0)
    if not np.any(mask):
        return float('inf')
    return float(np.min(rg[mask]))

# ================== TUNING ==================

SETPOINT   = 0.50   # target distance to wall (m)
KP         = 0.9    # normal P-gain
DEADBAND   = 0.04   # m around setpoint
V_TAN      = 0.22   # forward-along-wall speed (m/s)
V_MAX      = 0.45   # clamp |v|
KYAW       = 1.0    # yaw align to tangent
WZ_MAX     = 0.9

EMA_A      = 0.30   # smoothing for shortest distance
DT         = 0.07   # control rate
# Obstacle bypass
OBS_TRIG   = 0.45   # if something appears within this front distance -> AVOID
OBS_CLEAR  = 0.60   # when front is clear beyond this -> back to FOLLOW
AVOID_PUSH = 0.30   # m/s push AWAY from wall while avoiding
AVOID_TURN = 0.6    # extra yaw during avoid
AVOID_SLOW = 0.6    # scale down vx,vy during avoid

# ================== STATE & SMOOTHING ==================

_prev_d = None
mode = "FOLLOW"     # or "AVOID"

def smooth(d):
    global _prev_d
    if not math.isfinite(d): return d
    _prev_d = d if _prev_d is None else (EMA_A*d + (1.0-EMA_A)*_prev_d)
    return _prev_d

# ================== MAIN ==================

robot = MBot()
print(f"[INFO] Wall follower w/ obstacle bypass. Setpoint={SETPOINT:.2f} m")

try:
    while True:
        ranges, thetas = robot.read_lidar()

        # 1) shortest ray → wall normal
        d_min, th_min = find_min_dist(ranges, thetas)
        if not math.isfinite(d_min):
            robot.drive(0.0, 0.0, 0.0)
            time.sleep(DT)
            continue
        d_used = smooth(d_min)

        nx, ny = math.cos(th_min), math.sin(th_min)   # normal toward wall
        n_hat = [nx, ny, 0.0]

        # 2) tangent = n̂ × k̂
        tx, ty, _ = cross_product(n_hat, [0.0, 0.0, 1.0])  # = [ny, -nx, 0]
        if tx < 0.0:   # prefer forward-ish
            tx, ty = -tx, -ty
        tx, ty = unit2(tx, ty)
        nxu, nyu = unit2(nx, ny)

        # base along-wall command
        vx_follow = V_TAN * tx
        vy_follow = V_TAN * ty

        # base normal correction to hold setpoint
        err = SETPOINT - d_used
        corr = 0.0 if abs(err) <= DEADBAND else KP * err
        vx_corr = corr * nxu
        vy_corr = corr * nyu

        # desired yaw aligns with tangent, helps cornering
        desired_yaw = math.atan2(ty, tx)
        wz_align = clamp(KYAW * desired_yaw, -WZ_MAX, WZ_MAX)

        # 3) obstacle check and mode switch
        fwd = front_min_dist(ranges, thetas, fov_deg=24.0)
        if mode == "FOLLOW" and fwd < OBS_TRIG:
            mode = "AVOID"
        elif mode == "AVOID" and fwd > OBS_CLEAR:
            mode = "FOLLOW"

        if mode == "FOLLOW":
            vx = vx_follow + vx_corr
            vy = vy_follow + vy_corr
            wz = wz_align
        else:
            # AVOID: go AROUND object by moving AWAY from the wall (opposite normal),
            # keep some forward motion, and yaw a bit to curve around.
            # Opposite normal = (-nxu, -nyu)
            vx = (vx_follow + vx_corr - AVOID_PUSH * nxu) * AVOID_SLOW
            vy = (vy_follow + vy_corr - AVOID_PUSH * nyu) * AVOID_SLOW
            # steer slightly the same direction as tangent + extra
            wz = clamp(wz_align + AVOID_TURN * (1.0 if ny >= 0 else -1.0), -WZ_MAX, WZ_MAX)

        # clamp planar speed
        spd = math.hypot(vx, vy)
        if spd > V_MAX:
            s = V_MAX / spd
            vx, vy = vx*s, vy*s

        robot.drive(vx, vy, wz)
        time.sleep(DT)

except KeyboardInterrupt:
    pass
finally:
    try:
        robot.stop()
    except Exception:
        try: robot.drive(0.0, 0.0, 0.0)
        except Exception: pass
    print("[INFO] Stopped.")
