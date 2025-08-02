"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys
import math
import numpy
# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here

STATES = {1: "WALL_FOLLOW", 2: "ELEVATOR", 3: "LEFT_WALL_FOLLOW"}
ELEVATOR_STATES = {1: "APPROACHING", 2: "WAITING", 3: "AT_BOTTOM", 4: "ASCENDING", 5: "EXITING", 6: "DONE"}
current_state = "WALL_FOLLOW"
current_elevator_state = "APPROACHING"
########################################################################################
# Functions
########################################################################################

# this function returns area of a given ar tag using the corners and basic area formula
def find_area(corners):
    return abs((corners[2][0]-corners[0][0]) * (corners[2][1] - corners[0][1]))

#this function returns what ar tags the camera sees
def update_ar_tags():
    image = rc.camera.get_color_image()
    #get list of markers
    markers = rc_utils.get_ar_markers(image)
    #get id of first marker in list
    id = markers[0].get_id()
    #get area of said marker
    area = find_area(markers[0].get_corners())
    return id, area

#this function returns given angle but in positive form (but same location)
def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

#this function has no parameters but completes wall following calculations using lidar to return speed and angle.
#it's run in update() and by having it as its own function allows for better code organization
def wall_follow():
    scan_data = rc.lidar.get_samples()
    fov_span = 120
    #thesea are a bunch of tuned constants used later
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2
    #initialize these
    chosen_heading = 0
    best_opening = 0
    #this part iterates on each potential heading from -75 to 75, for the left wall follow version we change this window to
    #make it biased towards the left. we basically prevent it from seeing openings to the right
    for heading in range(-75, 75, step):
        # "fan out" from around the potential heading
        start = heading - fan_width // 2
        end = heading + fan_width // 2

        samples = []
        for ang in range(start, end + 1):
            
            adjusted = fix_angle(ang)
            dist = rc_utils.get_lidar_average_distance(scan_data, adjusted)
            if dist is not None and dist > scan_limit:
                samples.append(dist)

        if not samples or min(samples) < min_gap:
            continue

        candidate_clearance = min(samples)
        if candidate_clearance > best_opening:
            chosen_heading = heading
            best_opening = candidate_clearance

    special_light = 60
    sample_window = 2
    kp = 0.003

    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan_data, (0, 180))
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan_data, (180, 360))
    r_shift = rc_utils.get_lidar_average_distance(scan_data, special_light, sample_window)
    l_shift = rc_utils.get_lidar_average_distance(scan_data, 360 - special_light, sample_window)

    r_component = math.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = math.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    error = r_component - l_component
    wall_adjust = rc_utils.clamp(error * kp, -1, 1)

    merged_angle = rc_utils.clamp((chosen_heading / 30.0 + wall_adjust) / 2.0, -1.0, 1.0)

    # speed = 1.0 if best_opening > 220 else 0.6
    kp_s = 1
    speed = rc_utils.clamp(kp_s * (1 - abs(merged_angle)), 0.6, 1)
    print(f"speed: {speed}")
    return speed, merged_angle

#this function has no parameters but contains elevator state machine logic to return speed and angle 
#to ride the elevator
def ride_elevator(): #TODO: other team members put wtv inputs you want here
    return speed, angle #these need to be defined in this code

#this funciton uses the same base code as the wall_follow() function but it has slightly modified view angle so that 
#it is biased towards the left side. this allows us to get off of the elevator and ensure that we take the correct path to the left
def wall_follow_left():
    scan_data = rc.lidar.get_samples()
    fov_span = 120
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2

    chosen_heading = 0
    best_opening = 0

    for heading in range(-75, 20, step):
        start = heading - fan_width // 2
        end = heading + fan_width // 2

        samples = []
        for ang in range(start, end + 1):
            adjusted = fix_angle(ang)
            dist = rc_utils.get_lidar_average_distance(scan_data, adjusted)
            if dist is not None and dist > scan_limit:
                samples.append(dist)

        if not samples or min(samples) < min_gap:
            continue

        candidate_clearance = min(samples)
        if candidate_clearance > best_opening:
            chosen_heading = heading
            best_opening = candidate_clearance

    special_light = 60
    sample_window = 2
    kp = 0.003

    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan_data, (0, 180))
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan_data, (180, 360))
    r_shift = rc_utils.get_lidar_average_distance(scan_data, special_light, sample_window)
    l_shift = rc_utils.get_lidar_average_distance(scan_data, 360 - special_light, sample_window)

    r_component = math.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = math.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    error = r_component - l_component
    wall_adjust = rc_utils.clamp(error * kp, -1, 1)

    merged_angle = rc_utils.clamp((chosen_heading / 30.0 + wall_adjust) / 2.0, -1.0, 1.0)

    # speed = 1.0 if best_opening > 220 else 0.6
    kp_s = 1
    speed = rc_utils.clamp(kp_s * (1 - abs(merged_angle)), 0.6, 1)
    print(f"speed: {speed}")
    return speed, merged_angle

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle
    rc.drive.set_max_speed(1)
    speed = 1.0
    angle = 0.0
    rc.set_update_slow_time(0.5)

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global current_state, current_elevator_state
    #get the (hopefully only) ar tag it sees and set these variables
    marker_id, marker_area = update_ar_tags()
    #LOGIC TO DETERMINE THE STATE
    #if it sees marker 1 and it's close enough to be a certain area then set state to elevator
    if marker_id == 1 and marker_area >= 9000: #TODO: we can change this value depending on tuning
        current_state = "ELEVATOR"
    #the "DONE" elevator state is selected after we've wall followed down the ramp for a long enough time, and
    #this means that we are no longer doing the elevator so we can switch to what's immediately after, left wall following
    elif current_elevator_state == "DONE":
        current_state = "LEFT_WALL_FOLLOW"
    #we only need to left wall follow to make sure we select the correct path in the area with all the lines after the elevator.
    #this means that once we see ar tag 5 right before the tunnel and the S bend, we can switch back to regular wall following
    elif marker_id == 5 and marker_area >= 9000: #TODO: again, we will tune this number
        current_state = "WALL_FOLLOW"
    #LOGIC FOR WHAT TO DO DEPENDING ON WHAT STATE
    #these are lowkey pretty self explanatory, all these functions have built in controllers and data retrieval functions and
    #return what angle and speed the car should be at
    if current_state == "WALL_FOLLOW":
        speed, angle = wall_follow()
    elif current_state == "ELEVATOR":
        speed, angle = ride_elevator()
    else:
        speed, angle = wall_follow_left()
    #sets speed and angle based on what was returned
    rc.drive.set_speed_angle(speed, angle)
# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
