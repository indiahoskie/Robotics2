#!/usr/bin/env python3
import time, math, numpy as np
from mbot_bridge.api import MBot

# ---------- Required functions ----------
def find_min_dist(ranges, thetas):
    """Return (min_dist, min_angle) for the shortest VALID (range>0) ray."""
    if not ranges or not thetas or len(ranges) != len(thetas):
        return float('inf'), 0.0
    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    rg = np.clip(rg, 0.0, 8.0)           # ignore crazy-long returns
    valid = rg > 0.0
    if not np.any(valid):
        return float('inf'), 0.0
    idx_valid = np.nonzero(valid)[0]
    k = idx_valid[np.argmin(rg[valid])]
    return float(rg[k]), float(th[k])

def cross_product(v1, v2):
    """3D cross product (manual)."""
    x1,y1,z1 = v1; x2,y2,z2 = v2
    return [y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2]

# (optional) immediate call to show it works
_ = cross_product([1,0,0],[0,1,0])  # -> [0,0,1]

# ---------- Small helpers ----------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def unit2(x, y):
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < 1e-9 else (x/n, y/n)

# ---------- Tunables ----------
SETPOINT  = 0.50   # meters from wall
KP        = 0.9    # normal-distance P gain
DEADBAND  = 0.04   # m
V_TAN     = 0.22   # m/s along-wall
V_MAX     = 0.45   # m/s clamp for planar speed
KYAW      = 1.0    # yaw gain to align with wall tangent
WZ_MAX    = 0.9    # rad/s clamp
EMA_A     = 0.30   # smoothing for shortest distance
DT        = 0.07   # control period (~14 Hz)

# light smoothing so min-ray noise doesn’t cause thrash
_prev_d = None
def smooth(d):
    global _prev_d
    if not math.isfinite(d): return d
    _prev_d = d if _prev_d is None else (EMA_A*d + (1-EMA_A)*_prev_d)
    return _prev_d

# optional: if something is straight ahead, ease speed & add turn bias to take corners
def front_distance(ranges, thetas):
    rg = np.asarray(ranges, float); th = np.asarray(thetas, float)
    th = (th + math.pi) % (2*math.pi) - math.pi
    mask = (np.abs(th) <= math.radians(12.0)) & (rg > 0.0)
    return float(np.min(rg[mask])) if np.any(mask) else float('inf')

# ---------- Main ----------
robot = MBot()
print(f"[INFO] Wall follower (shortest ray + cross product). Setpoint={SETPOINT:.2f} m")

try:
    while True:
        ranges, thetas = robot.read_lidar()

        # 1) shortest ray -> distance & angle
        d_min, th_min = find_min_dist(ranges, thetas)
        if not math.isfinite(d_min):
            robot.drive(0.0, 0.0, 0.0)
            time.sleep(DT)
            continue

        d_used = smooth(d_min)

        # 2) wall normal n̂ from shortest ray
        nx, ny = math.cos(th_min), math.sin(th_min)
        n_hat = [nx, ny, 0.0]

        # 3) wall tangent t̂ = n̂ × k̂
        tx, ty, _ = cross_product(n_hat, [0.0, 0.0, 1.0])   # -> [ny, -nx, 0]
        # prefer forward-ish tangent (positive x projection)
        if tx < 0.0: tx, ty = -tx, -ty

        # normalize
        tx, ty = unit2(tx, ty)
        nxu, nyu = unit2(nx, ny)

        # 4) distance error (+ if too far from wall)
        err = SETPOINT - d_used
        corr = 0.0 if abs(err) <= DEADBAND else KP * err

        # 5) planar velocity: follow wall + hold distance
        vx = V_TAN * tx + corr * nxu
        vy = V_TAN * ty + corr * nyu
        spd = math.hypot(vx, vy)
        if spd > V_MAX:
            s = V_MAX / spd
            vx, vy = vx*s, vy*s

        # 6) yaw to align with the wall so it TURNS AT CORNERS
        desired_heading = math.atan2(ty, tx)   # angle of tangent in body frame
        wz = clamp(KYAW * desired_heading, -WZ_MAX, WZ_MAX)

        # 7) corner assist: if front is getting close, slow forward and add turn
        fd = front_distance(ranges, thetas)
        if fd < 0.45:     # start easing before we’re too close
            vx *= 0.6; vy *= 0.6
            # small extra turn toward the wall
            wz = clamp(wz + (0.35 if nx > 0 else -0.35), -WZ_MAX, WZ_MAX)

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
    print("[INFO] Robot stopped.")






