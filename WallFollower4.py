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

def clamp(x, lo, hi): 
    return lo if x < lo else hi if x > hi else x

def unit2(x, y):
    """Normalize 2D vector to unit length"""
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < 1e-9 else (x/n, y/n)

def wrap_pi(a): 
    """Wrap angle to [-pi, pi]"""
    return (a + math.pi) % (2*math.pi) - math.pi

def front_min_dist(ranges, thetas, fov_deg=30.0):
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

def side_bias_dist(ranges, thetas, side='left', fov_deg=60.0):
    """Get minimum distance on a specific side (for wall detection)"""
    if not ranges or not thetas:
        return float('inf')
    rg = np.asarray(ranges, float)
    th = np.asarray(thetas, float)
    th = (th + math.pi) % (2*math.pi) - math.pi
    
    if side == 'left':
        # Left side: angles between 30° and 150° (0.52 to 2.62 rad)
        mask = (th > math.radians(30)) & (th < math.radians(150)) & (rg > 0.0)
    else:  # right
        # Right side: angles between -150° and -30°
        mask = (th < math.radians(-30)) & (th > math.radians(-150)) & (rg > 0.0)
    
    if not np.any(mask):
        return float('inf')
    return float(np.min(rg[mask]))

# ================== TUNING ==================

SETPOINT   = 0.50   # target distance to wall (m)
KP         = 1.2    # P-gain for wall distance correction
DEADBAND   = 0.03   # m around setpoint (no correction zone)
V_TAN      = 0.30   # speed along the wall (m/s)
V_MAX      = 0.50   # clamp total velocity
KYAW       = 0.8    # yaw alignment gain
WZ_MAX     = 1.0    # max angular velocity

EMA_A      = 0.40   # smoothing for distance measurements
DT         = 0.05   # control loop period (s)

# Obstacle bypass parameters
OBS_TRIG   = 0.40   # if front obstacle closer than this -> start AVOID
OBS_CLEAR  = 0.70   # when front clears beyond this -> return to FOLLOW
AVOID_LATERAL = 0.35  # lateral speed to move away from wall during avoid (m/s)
AVOID_FORWARD = 0.25  # forward speed during avoid (m/s)
AVOID_YAW  = 0.7    # additional yaw rate during avoidance

# ================== STATE & SMOOTHING ==================

_prev_d = None
_prev_front = None
mode = "FOLLOW"     # States: "FOLLOW" or "AVOID"
avoid_timer = 0.0   # Time spent in avoid mode

def smooth(d, prev, alpha=EMA_A):
    """Exponential moving average"""
    if not math.isfinite(d): 
        return prev if prev is not None else d
    if prev is None:
        return d
    return alpha * d + (1.0 - alpha) * prev

# ================== MAIN ==================

robot = MBot()
print(f"[INFO] Wall Follower with Obstacle Avoidance")
print(f"[INFO] Setpoint: {SETPOINT:.2f}m | Tangent speed: {V_TAN:.2f} m/s")
print(f"[INFO] Obstacle trigger: {OBS_TRIG:.2f}m | Clear threshold: {OBS_CLEAR:.2f}m")

try:
    while True:
        loop_start = time.time()
        
        # ---- READ LIDAR ----
        try:
            ranges, thetas = robot.read_lidar()
        except Exception as e:
            print(f"[ERROR] Lidar read failed: {e}")
            robot.drive(0.0, 0.0, 0.0)
            time.sleep(DT)
            continue

        # ---- 1) FIND NEAREST WALL ----
        d_min, th_min = find_min_dist(ranges, thetas)
        
        if not math.isfinite(d_min):
            print("[WARN] No valid wall detected - stopping")
            robot.drive(0.0, 0.0, 0.0)
            time.sleep(DT)
            continue
        
        # Smooth the distance measurement
        d_used = smooth(d_min, _prev_d)
        _prev_d = d_used

        # ---- 2) COMPUTE WALL NORMAL AND TANGENT ----
        # Normal vector points FROM robot TOWARD wall
        nx, ny = math.cos(th_min), math.sin(th_min)
        n_hat = [nx, ny, 0.0]

        # Tangent = normal × z-axis (perpendicular to normal in xy-plane)
        # This gives us the direction along the wall
        tx, ty, _ = cross_product(n_hat, [0.0, 0.0, 1.0])  # = [ny, -nx, 0]
        
        # Ensure tangent points generally forward (prefer positive x-component)
        if tx < 0.0:
            tx, ty = -tx, -ty
        
        tx, ty = unit2(tx, ty)
        nxu, nyu = unit2(nx, ny)

        # ---- 3) WALL FOLLOWING CONTROL ----
        # Base velocity: move along wall tangent
        vx_tangent = V_TAN * tx
        vy_tangent = V_TAN * ty

        # Correction: maintain setpoint distance from wall
        # Error is positive when too far from wall (need to move toward wall)
        # Error is negative when too close to wall (need to move away from wall)
        err = SETPOINT - d_used
        
        # Apply deadband to prevent oscillation
        if abs(err) <= DEADBAND:
            corr = 0.0
        else:
            # Negative error means we're too close, so we move AWAY (opposite of normal)
            # Positive error means we're too far, so we move TOWARD (along normal)
            corr = -KP * err  # Negative sign: reduce distance by moving toward wall
        
        vx_correction = corr * nxu
        vy_correction = corr * nyu

        # Desired yaw: align robot heading with tangent direction
        desired_yaw = math.atan2(ty, tx)
        wz_align = clamp(KYAW * desired_yaw, -WZ_MAX, WZ_MAX)

        # ---- 4) OBSTACLE DETECTION ----
        fwd_dist = front_min_dist(ranges, thetas, fov_deg=35.0)
        fwd_dist = smooth(fwd_dist, _prev_front, alpha=0.5)
        _prev_front = fwd_dist

        # ---- 5) STATE MACHINE ----
        prev_mode = mode
        
        if mode == "FOLLOW":
            if fwd_dist < OBS_TRIG:
                mode = "AVOID"
                avoid_timer = 0.0
                print(f"[STATE] FOLLOW -> AVOID (obstacle at {fwd_dist:.2f}m)")
                
        elif mode == "AVOID":
            avoid_timer += DT
            
            # Return to follow if:
            # 1. Front is clear AND
            # 2. We've been avoiding for at least 0.5 seconds (prevent rapid switching)
            if fwd_dist > OBS_CLEAR and avoid_timer > 0.5:
                mode = "FOLLOW"
                print(f"[STATE] AVOID -> FOLLOW (cleared at {fwd_dist:.2f}m)")

        # ---- 6) GENERATE CONTROL COMMANDS ----
        if mode == "FOLLOW":
            # Normal wall following
            vx = vx_tangent + vx_correction
            vy = vy_tangent + vy_correction
            wz = wz_align
            
            status = f"FOLLOW | wall={d_used:.2f}m | front={fwd_dist:.2f}m | err={err:+.3f}m"
            
        else:  # AVOID
            # Obstacle avoidance: move AWAY from wall while going forward
            # Strategy: swing wide around the obstacle
            
            # Move laterally away from wall (opposite of normal direction)
            vx_lateral = -AVOID_LATERAL * nxu
            vy_lateral = -AVOID_LATERAL * nyu
            
            # Keep some forward motion along tangent
            vx_forward = AVOID_FORWARD * tx
            vy_forward = AVOID_FORWARD * ty
            
            # Combine: move away from wall + move forward
            vx = vx_forward + vx_lateral
            vy = vy_forward + vy_lateral
            
            # Add steering to curve around obstacle
            # Turn in the direction away from the wall
            turn_sign = 1.0 if ny >= 0 else -1.0  # Turn left if wall is on left
            wz = clamp(wz_align + AVOID_YAW * turn_sign, -WZ_MAX, WZ_MAX)
            
            status = f"AVOID  | wall={d_used:.2f}m | front={fwd_dist:.2f}m | t={avoid_timer:.1f}s"

        # ---- 7) VELOCITY LIMITING ----
        # Clamp total planar speed
        spd = math.hypot(vx, vy)
        if spd > V_MAX:
            scale = V_MAX / spd
            vx *= scale
            vy *= scale

        # ---- 8) SEND COMMANDS ----
        robot.drive(vx, vy, wz)
        
        print(f"[{status}]")
        print(f"  cmd: vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f}")

        # Maintain loop rate
        elapsed = time.time() - loop_start
        sleep_time = DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Keyboard interrupt - stopping")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        robot.stop()
    except Exception:
        try: 
            robot.drive(0.0, 0.0, 0.0)
        except Exception: 
            pass
    print("[INFO] Robot stopped.")
