#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) - Improved Version
- Finds the nearest obstacle (ignoring zero ranges)
- Moves the robot laterally toward that obstacle's direction
- Maintains a setpoint distance using P-control
- Does NOT spin: wz is held at 0.0 (pure translation in x/y)
- Moves in whatever direction the obstacle is located

Expected behavior:
- Robot follows the nearest obstacle/human as they move around
- Can move in multiple directions in the plane (2D) without rotating to face the target
- Follows the shortest angle/distance continuously
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ---------------------------- CONFIG ----------------------------
SETPOINT_M      = 0.75     # desired distance to target (m)
TOLERANCE_M     = 0.05     # acceptable error band around setpoint (m)

KP_DIST         = 1.2      # P gain for radial distance control (m/s per meter error)

V_MAX           = 0.55     # absolute cap on translational speed (m/s)
V_MIN           = 0.08     # small floor to overcome static friction when moving (m/s)
DEADBAND_M      = 0.03     # small deadband near zero error

# Reduced smoothing for faster response to direction changes
SMOOTH_ALPHA    = 0.6      # EMA smoothing for theta_min (higher = faster response)

SCAN_TIMEOUT_S  = 0.25     # if scans lag beyond this, stop for safety
LOOP_HZ         = 20       # control loop rate (increased for better response)

PRINT_DEBUG     = True
# -----------------------------------------------------------------

def clamp(x, lo, hi):
    """Clamp value x between lo and hi"""
    return lo if x < lo else hi if x > hi else x

def ema(prev, new, alpha):
    """Exponential moving average for smooth transitions"""
    return alpha * new + (1.0 - alpha) * prev

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def find_min_nonzero(ranges, thetas):
    """
    Return (min_distance, angle_at_min) ignoring zero/invalid ranges.
    If nothing valid, returns (np.inf, 0.0)
    
    This implements findMinNonzeroDist() behavior:
    - Ignores ranges that are 0.0 (bad lidar readings)
    - Returns the shortest valid distance and its corresponding angle
    """
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    
    # Mask out zero and invalid ranges
    mask = r > 0.0
    
    if not np.any(mask):
        return (np.inf, 0.0)
    
    r_valid = r[mask]
    t_valid = t[mask]
    
    # Find minimum distance
    idx = np.argmin(r_valid)
    
    return float(r_valid[idx]), float(t_valid[idx])

class Driver:
    """
    Motion adapter for 2D planar movement without rotation.
    Handles different MBot API versions gracefully.
    """
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None

        if hasattr(bot, "drive"):
            sig = inspect.signature(bot.drive)
            # Prefer 3-parameter drive for full 2D control
            if len(sig.parameters) >= 3:
                self.mode = "drive_vx_vy_wz"  # Ideal: (vx, vy, wz)
            elif len(sig.parameters) == 1:
                self.mode = "drive_vx"        # Limited: (vx) only
            else:
                self.mode = "drive_generic"
        elif hasattr(bot, "set_velocity"):
            self.mode = "set_velocity"        # (vx, wz) - no lateral
        elif hasattr(bot, "motors"):
            self.mode = "motors"              # Low-level differential
        else:
            raise RuntimeError("No valid drive function found on MBot")

        if PRINT_DEBUG:
            print(f"[Driver] Using mode: {self.mode}")

    def send(self, vx, vy=0.0, wz=0.0):
        """
        Send velocity commands to robot.
        Always forces wz=0.0 to prevent spinning.
        """
        # CRITICAL: No spinning allowed in Follow 2D
        wz = 0.0

        if self.mode == "drive_vx_vy_wz":
            # Full 2D control - this is what we want!
            self.bot.drive(vx, vy, wz)
            
        elif self.mode == "drive_vx":
            # Limited: only forward/backward
            # Project 2D velocity into x direction
            v_total = math.sqrt(vx**2 + vy**2)
            if vx < 0:
                v_total = -v_total
            self.bot.drive(v_total)
            
        elif self.mode == "set_velocity":
            # Can't command vy, only vx
            self.bot.set_velocity(vx, 0.0)
            
        elif self.mode == "motors":
            # Differential drive approximation
            # Both wheels same speed = straight motion
            self.bot.motors(vx, vx)

    def stop(self):
        """Safely stop the robot"""
        if hasattr(self.bot, "stop"):
            self.bot.stop()
        else:
            self.send(0.0, 0.0, 0.0)

def main():
    bot = MBot()
    driver = Driver(bot)

    # Smoothed theta for stable direction tracking
    theta_smooth = None  # Start with None to initialize on first reading
    last_scan_time = time.time()

    print("[INFO] Follow-Me 2D started (lateral movement, no spinning).")
    print(f"[INFO] Setpoint: {SETPOINT_M}m, P-gain: {KP_DIST}, Max speed: {V_MAX} m/s")
    print("[INFO] Robot will follow nearest obstacle in 2D without rotating.")

    try:
        while True:
            loop_start = time.time()
            
            # ---- READ LIDAR ----
            # ranges: distances in meters (0.0 = invalid)
            # thetas: angles in radians (0 = forward, +pi/2 = left, -pi/2 = right)
            try:
                ranges, thetas = bot.read_lidar()
            except Exception as e:
                print(f"[ERROR] Failed to read lidar: {e}")
                driver.stop()
                time.sleep(0.1)
                continue
            
            now = time.time()
            
            # Safety: check for scan timeout
            if now - last_scan_time > SCAN_TIMEOUT_S:
                if PRINT_DEBUG:
                    print("[WARN] Lidar scan timeout - stopping for safety")
                driver.send(0.0, 0.0, 0.0)
                
            last_scan_time = now

            # ---- FIND NEAREST VALID OBSTACLE ----
            # This implements the "find minimum non-zero distance" requirement
            d_min, th_min = find_min_nonzero(ranges, thetas)

            if PRINT_DEBUG:
                if np.isfinite(d_min):
                    print(f"[SCAN] Nearest obstacle: {d_min:.3f}m at {math.degrees(th_min):+.1f}°")
                else:
                    print("[SCAN] No valid obstacles detected (all zero/invalid).")

            # ---- HANDLE NO TARGET ----
            if not np.isfinite(d_min):
                driver.send(0.0, 0.0, 0.0)
                theta_smooth = None  # Reset smoothing
                time.sleep(1.0 / LOOP_HZ)
                continue

            # ---- SMOOTH DIRECTION FOR STABILITY ----
            # Initialize smoothing on first valid reading
            if theta_smooth is None:
                theta_smooth = th_min
            else:
                # Handle angle wrapping for smooth transitions
                angle_diff = normalize_angle(th_min - theta_smooth)
                theta_smooth = normalize_angle(theta_smooth + SMOOTH_ALPHA * angle_diff)

            # ---- RADIAL P-CONTROL ON DISTANCE ----
            # Error: positive = too far (move toward), negative = too close (back away)
            # We want to REDUCE the error, so we move in the OPPOSITE direction of error
            err = d_min - SETPOINT_M

            # Apply deadband to prevent jitter when at setpoint
            if abs(err) < DEADBAND_M:
                vx_cmd = 0.0
                vy_cmd = 0.0
                
                if PRINT_DEBUG:
                    print(f"[CTRL] At setpoint (err={err:+.3f}m) - holding position")
                    
            else:
                # P-control: velocity magnitude proportional to distance error
                # NEGATIVE sign because we want to move TOWARD obstacle (reduce distance)
                v_mag = -KP_DIST * err

                # Apply velocity limits
                v_sign = 1.0 if v_mag >= 0 else -1.0
                v_mag = abs(v_mag)
                v_mag = clamp(v_mag, 0.0, V_MAX)
                
                # Apply minimum velocity to overcome static friction
                if v_mag > 0.0:
                    v_mag = max(v_mag, V_MIN)
                
                # Restore sign
                v_mag *= v_sign

                # ---- CONVERT POLAR TO CARTESIAN ----
                # Move in the direction of the obstacle (theta_smooth)
                # Robot frame: +x forward, +y left
                vx_cmd = v_mag * math.cos(theta_smooth)
                vy_cmd = v_mag * math.sin(theta_smooth)
                
                if PRINT_DEBUG:
                    print(f"[CTRL] err={err:+.3f}m → v_mag={v_mag:+.3f} m/s in direction {math.degrees(theta_smooth):+.1f}°")

            # ---- SEND COMMANDS (NO ROTATION) ----
            driver.send(vx_cmd, vy_cmd, 0.0)

            if PRINT_DEBUG:
                print(f"[CMD]  vx={vx_cmd:+.3f} m/s, vy={vy_cmd:+.3f} m/s, wz=+0.000 rad/s")
                print("-" * 60)

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / LOOP_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received - stopping robot.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely. Program terminated.")

if __name__ == "__main__":
    main()
