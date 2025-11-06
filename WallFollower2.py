import time

# Example sensor and motor control functions
def get_distance_from_wall():
    # Replace with actual sensor reading logic
    return 20  # Example: distance in cm

def set_motor_speeds(left_speed, right_speed):
    # Replace with actual motor control logic
    print(f"Left motor: {left_speed}, Right motor: {right_speed}")

# Wall-following parameters
TARGET_DISTANCE = 15  # Target distance from the wall in cm
SPEED = 50            # Base speed for motors
KP = 1.5              # Proportional gain for correction

def wall_follower():
    while True:
        # Get the current distance from the wall
        distance = get_distance_from_wall()
        
        # Calculate the error (difference from target distance)
        error = TARGET_DISTANCE - distance
        
        # Calculate correction using proportional control
        correction = KP * error
        
        # Adjust motor speeds based on correction
        left_speed = SPEED - correction
        right_speed = SPEED + correction
        
        # Set motor speeds
        set_motor_speeds(left_speed, right_speed)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)

# Run the wall follower
try:
    wall_follower()
except KeyboardInterrupt:
    print("Wall follower stopped.")
