import time
import numpy as np
from mbot_bridge.api import MBot

# Get distance to wall
def find_fwd_dist(ranges, thetas, window=5):
    fwd_ranges = np.array(ranges[:window] + ranges[-window:])
    fwd_thetas = np.array(thetas[:window] + thetas[-window:])
    valid_idx = (fwd_ranges > 0).nonzero()
    fwd_ranges = fwd_ranges[valid_idx]
    fwd_thetas = fwd_thetas[valid_idx]
    fwd_dists = fwd_ranges * np.cos(fwd_thetas)
    return np.mean(fwd_dists)

# Initialize robot
robot = MBot()

# Desired distance to wall (in meters)
setpoint = 0.5  # You can adjust this value

# Proportional gain
Kp = 1.0  # Tune this value for responsiveness

try:
    while True:
        # Read Lidar
        ranges, thetas = robot.read_lidar()

        # Measure distance to wall
        dist_to_wall = find_fwd_dist(ranges, thetas)

        # Compute error
        error = setpoint - dist_to_wall

        # Compute velocity command
        velocity = Kp * error

        # Clamp velocity to safe limits
        velocity = max(min(velocity, 0.5), -0.5)  # Limits between -0.5 and 0.5 m/s

        # Send velocity command to robot
        robot.set_velocity(linear=velocity, angular=0.0)

        # Wait before next scan
        time.sleep(0.1)

except:
    robot.stop()
