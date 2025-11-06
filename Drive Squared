#!/usr/bin/env python3
"""
P1.1: Drive a 1m x 1m square three times (MBot-Omni).
Uses timed velocity commands; not expected to be perfect.
"""

import time
from mbot_bridge.api import MBot

# Tune these for your robot / floor
V = 0.25          # m/s magnitude
SIDE_TIME = 4.0   # seconds per 1m side at V (adjust to hit ~1m)
PAUSE = 0.2       # small pause between segments
LAPS = 3

def main():
    bot = MBot()
    print("[INFO] Driving a square 3 times... (Ctrl-C to stop)")
    try:
        for lap in range(LAPS):
            print(f"[INFO] Lap {lap+1}/{LAPS}")
            # forward
            bot.drive(V, 0.0, 0.0); time.sleep(SIDE_TIME); bot.drive(0,0,0); time.sleep(PAUSE)
            # left
            bot.drive(0.0, V, 0.0); time.sleep(SIDE_TIME); bot.drive(0,0,0); time.sleep(PAUSE)
            # backward
            bot.drive(-V, 0.0, 0.0); time.sleep(SIDE_TIME); bot.drive(0,0,0); time.sleep(PAUSE)
            # right
            bot.drive(0.0, -V, 0.0); time.sleep(SIDE_TIME); bot.drive(0,0,0); time.sleep(PAUSE)

        bot.drive(0.0, 0.0, 0.0)
        print("[INFO] Done.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        try:
            bot.drive(0.0, 0.0, 0.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
