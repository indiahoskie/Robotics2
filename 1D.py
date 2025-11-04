from mbot import Mbot

# Initialize the mBot
robot = Mbot()

# Function to move forward
def move_forward(speed, duration):
    robot.set_motor_speed(left_speed=speed, right_speed=speed)
    robot.sleep(duration)
    robot.stop()

# Function to move backward
def move_backward(speed, duration):
    robot.set_motor_speed(left_speed=-speed, right_speed=-speed)
    robot.sleep(duration)
    robot.stop()

# Main program
try:
    print("Moving forward...")
    move_forward(speed=100, duration=2)  # Move forward at speed 100 for 2 seconds

    print("Moving backward...")
    move_backward(speed=100, duration=2)  # Move backward at speed 100 for 2 seconds

    print("Done!")
except KeyboardInterrupt:
    print("Program interrupted!")
    robot.stop()
