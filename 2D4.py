#!/usr/bin/env python3
"""
Follow Me 2D (LiDAR) — Straight-Line for Unknown MBot API
- Autodetects available motion API and uses the best matching control:
    * drive(vx, wz)            -> diff-drive align (turn) then dash straight
    * set_velocity(vx, wz)     -> same as above
    * drive(vx, vy, wz)        -> holonomic: we still do rotate-then-dash to ensure straight lines
    * turn(wz) + drive(vx)     -> separate yaw & forward methods
- If NO rotation method is available, we abort gracefully (can't align without yaw control).
"""

import time
import math
import numpy as np
import inspect
from mbot_bridge.api import MBot

# ============================== CONFIG ==============================
# Forward straight motion
VX_MAX        = 0.35      # forward top speed (m/s)
VX_MIN        = 0.05
KX            = 1.2       # forward speed ~ KX * distance, clamped

# Turn-in-place alignment
WZ_MAX        = 1.0       # max turn speed (rad/s)
KW            = 3.0       # turn rate ~ KW * theta, clamped
TH_ALIGN      = 0.10      # aligned when |theta| < TH_ALIGN radians (~6 deg)

# Stop very near target
NEAR_STOP     = 0.06      # meters

# Loop & logging
LOOP_HZ       = 30
PRINT_DEBUG   = True
# ====================================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def find_min_nonzero(ranges, thetas):
    """Return (min_distance, angle_at_min) ignoring zero ranges; (inf, 0.0) if none."""
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    mask = r > 0.0
    if not np.any(mask):
        return (float("inf"), 0.0)
    r_valid = r[mask]
    t_valid = t[mask]
    i = int(np.argmin(r_valid))
    return float(r_valid[i]), float(t_valid[i])

class Driver:
    """
    Motion adapter that probes the MBot API and exposes:
        - send(vx, wz)  for diff-drive style control
        - send_holo(vx, vy, wz) if full holonomic is available
        - stop()
    Internally, it will use one of:
        * drive(vx, wz)
        * set_velocity(vx, wz)
        * drive(vx, vy, wz)
        * turn(wz) + drive(vx)
    """
    def __init__(self, bot: MBot):
        self.bot = bot
        self.mode = None

        # Try drive(...) and inspect its arity
        if hasattr(bot, "drive"):
            try:
                sig = inspect.signature(bot.drive)
                n = len(sig.parameters)
                # Common patterns we support:
                if n >= 3:
                    self.mode = "drive_vx_vy_wz"   # holonomic
                elif n == 2:
                    self.mode = "drive_vx_wz"      # diff-drive (ideal)
                elif n == 1:
                    # drive(vx) only — keep looking for a separate turn(...)
                    self.mode = "drive_vx_only"
                else:
                    self.mode = "drive_unknown"
            except Exception:
                self.mode = "drive_unknown"

        # If not settled on a usable mode yet, try set_velocity(vx, wz)
        if self.mode in (None, "drive_unknown", "drive_vx_only"):
            if hasattr(bot, "set_velocity"):
                try:
                    sig = inspect.signature(bot.set_velocity)
                    if len(sig.parameters) >= 2:
                        self.mode = "set_velocity_vx_wz"
                except Exception:
                    pass

        # If we only have drive(vx), check for a dedicated turn(wz)-style method
        self.turn_method = None
        if self.mode == "drive_vx_only":
            for name in ("turn", "rotate", "spin", "set_angular_velocity", "setYawRate", "yaw_rate"):
                if hasattr(bot, name):
                    self.turn_method = getattr(bot, name)
                    self.mode = "drive_vx__plus_turn"   # forward via drive(vx), yaw via turn(wz)
                    break

        if PRINT_DEBUG:
            print(f"[Driver] Detected mode: {self.mode}")

        # Final sanity: if we still don't have any way to command yaw, warn
        self.can_rotate = self.mode in ("drive_vx_wz", "set_velocity_vx_wz", "drive_vx_vy_wz", "drive_vx__plus_turn")

    def send(self, vx, wz):
        """
        Send diff-drive style command. If only holonomic is available, we set vy=0.
        """
        if self.mode == "drive_vx_wz":
            self.bot.drive(vx, wz)
        elif self.mode == "set_velocity_vx_wz":
            self.bot.set_velocity(vx, wz)
        elif self.mode == "drive_vx_vy_wz":
            # Use vy=0 to ensure straight lines (we rotate only during align)
            self.bot.drive(vx, 0.0, wz)
        elif self.mode == "drive_vx__plus_turn":
            # Issue yaw first (small dt), then forward
            # Here we assume the turn method takes (wz) in rad/s; many APIs also accept (wz) directly.
            try:
                self.turn_method(wz)
            except TypeError:
                # Some APIs want (duration, speed) or different sigs; if this happens, fall back to zero turn.
                pass
            self.bot.drive(vx)
        elif self.mode == "drive_vx_only":
            # No yaw capability — cannot align, only straight. We choose to stop for safety.
            self.bot.drive(0.0)
            raise RuntimeError("No rotation method available (drive(vx) only) — cannot align to target.")
        else:
            raise RuntimeError("No supported drive mode detected on MBot.")

    def stop(self):
        try:
            if hasattr(self.bot, "stop"):
                self.bot.stop()
            elif hasattr(self.bot, "drive"):
                # Best-effort stop
                sig = inspect.signature(self.bot.drive)
                n = len(sig.parameters)
                if n >= 3:
                    self.bot.drive(0.0, 0.0, 0.0)
                elif n == 2:
                    self.bot.drive(0.0, 0.0)
                elif n == 1:
                    self.bot.drive(0.0)
        except Exception:
            pass

def main():
    bot = MBot()
    driver = Driver(bot)

    if not driver.can_rotate:
        print("[ERROR] This MBot API exposes no rotation control (only drive(vx)).")
        print("        Without yaw control, the robot cannot align to the target; exiting safely.")
        return

    print("[INFO] Follow-Me Straight-Line (API-adaptive) started.")

    try:
        dt = 1.0 / LOOP_HZ
        while True:
            # --- Read LiDAR ---
            ranges, thetas = bot.read_lidar()
            d, th = find_min_nonzero(ranges, thetas)

            if not np.isfinite(d):
                driver.stop()
                time.sleep(dt)
                continue

            # --- Align: rotate until object is centered ---
            if abs(th) > TH_ALIGN:
                vx_cmd = 0.0
                wz_cmd = clamp(KW * th, -WZ_MAX, WZ_MAX)
            else:
                # --- Dash: go straight forward ---
                if d < NEAR_STOP:
                    vx_cmd = 0.0
                else:
                    vx_cmd = clamp(KX * d, VX_MIN, VX_MAX)
                wz_cmd = 0.0

            # --- Send to robot (API-adaptive) ---
            driver.send(vx_cmd, wz_cmd)

            if PRINT_DEBUG:
                print(f"[CMD] vx={vx_cmd:+.3f} m/s, wz={wz_cmd:+.3f} rad/s | d={d:.3f} m, th={th:+.3f} rad")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.stop()
        print("[INFO] Robot stopped safely.")

if __name__ == "__main__":
    main()


