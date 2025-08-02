# STATES:
# WALL FOLLOW: normal wall following state
# ELEVATOR: little state machine of its own, goes through the full elevator dynamic obstacle
# HARD LEFT: coming out of the elevator, take the hard left to skip big line follow thing
# LEFT WALL FOLLOW: wall follow with a left bias so we dont go in circles

# ELEVATOR STATES:
# 1) Approaching: approaching the elevator until we're a certain distance away
# 2) Check the light: if its green, go, if its red, wait
# 3) Waiting to see when it turns red
# 4) Counter 7 seconds until top
# 5) Check to see if bbox, if so, go back to state 3 cuz we messed up, else go to state 6
# 6) Exiting: get off


########################################################################################
# Imports
########################################################################################

import sys
import math
import numpy

import cv2
import os
import time

sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

default_path = 'models' # location of model weights and labels
model_name = 'elevator_edgetpu.tflite'
label_name = 'elevator_labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here

STATES = {1: "WALL_FOLLOW", 2: "ELEVATOR", 3: "LEFT_WALL_FOLLOW", 4: "HARD_LEFT"}
ELEVATOR_STATES = {1: "APPROACHING", 2: "WAITING", 3: "AT_BOTTOM", 4: "ASCENDING", 5: "EXITING", 6: "DONE"}
current_state = "WALL_FOLLOW"
elevator_state = "APPROACHING"

# Constant variables: time variables are for counters
# TODO: tune each of these variable
MIN_SPEED = 0.65
FORWARD_TIME = 2
ASCENSION_TIME = 6.5
RAMP_TIME = 1
LEFT_TIME = 2

# counters, contour color for elevator
contour_color = ""
ascension_counter = 0
forward_counter = 0
ramp_counter = 0
left_counter = 0

last_angle = 0

# TODO: tune each of these variables
GREEN = ((30, 50, 50), (80, 255, 255))
RED = ((165, 50, 50), (20, 255, 255))
MIN_CONTOUR_AREA = 1000
MIN_AR_AREA = 1000

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3

# loading the model pretty much
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

########################################################################################
# Functions
########################################################################################
# THE DICTIONARY
# the special characters
blank = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
])
plus = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
])
minus = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])
dot = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
])

# numbers
one = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
])
two = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1],
])
three = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])
four = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
])
five = np.array([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
])
six = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])
seven = np.array([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
])
eight = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])
nine = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])
zero = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
])

nums = {
    "1": one,
    "2": two,
    "3": three,
    "4": four,
    "5": five,
    "6": six,
    "7": seven,
    "8": eight,
    "9": nine,
    "0": zero,
}

chars = {
    "+": plus,
    "-": minus,
    ".": dot,
}

def insert_blank(dot_matrix):
    for i in range(len(dot_matrix)):
        dot_matrix[i] = np.concatenate([dot_matrix[i], blank[i]])

def insert_chars(dot_matrix, char):
    for i in range(len(dot_matrix)):
        dot_matrix[i] = np.concatenate([dot_matrix[i], chars[f"{char}"][i]])

def insert_nums(dot_matrix, num):
    for i in range(len(dot_matrix)):
        dot_matrix[i] = np.concatenate([dot_matrix[i], nums[f"{num}"][i]])

# update contour used only for elevator once we are on to detect the elevator state
def update_contour():
    global contour_color

    contour = None

    image = rc.camera.get_color_image()

    if image is None:
        contour_point = None
        contour_area = 0
    else:
        # TODO Part 2: Search for line colors, and update the global variables
        # contour_point and contour_area with the largest contour found
        green_contours = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        red_contours = rc_utils.find_contours(image, RED[0], RED[1])

        green = rc_utils.get_largest_contour(green_contours, MIN_CONTOUR_AREA)
        red = rc_utils.get_largest_contour(red_contours, MIN_CONTOUR_AREA)

        if red is not None:
            contour = red
            contour_color = "RED"
        elif green is not None:
            contour = green
            contour_color = "GREEN"

        if contour is not None:
            contour_point = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_point)

        # Display the image to the screen
    rc.display.show_color_image(image)

def get_obj_and_type(cv2_im, inference_size, objs):
    height, width, _ = cv2_im.shape
    max_score = 0
    correct_obj = None
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.score > max_score:
            max_score = obj.score
            correct_obj = obj
    bbox = correct_obj.bbox.scale(scale_x, scale_y)
    x0, y0 = int(bbox.xmin), int(bbox.ymin)
    x1, y1 = int(bbox.xmax), int(bbox.ymax)
    center = ((x0+x1)/2, (y0+y1)/2)
    id = correct_obj.id
    return center, id, correct_obj

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

    # if no good openings, uses triangles to find the biggest opening and use taht as the controller
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

    # speed controller based on the absolute value of the angle to slooowwww down around those turns
    kp_s = 1
    speed = rc_utils.clamp(kp_s * (1 - abs(merged_angle)), 0.6, 1)
    print(f"speed: {speed}")
    return speed, merged_angle

#this function has no parameters but contains elevator state machine logic to return speed and angle 
#to ride the elevator
def ride_elevator(): #TODO: other team members put wtv inputs you want here - DONE
    global speed, angle, last_angle, ascension_counter, forward_counter, elevator_state, current_state

    # the model stuff
    image = rc.camera.get_color_image()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, inference_size)
    run_inference(interpreter, rgb_image.tobytes())
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]
    center, id, obj = get_obj_and_type(image, inference_size, objs)

    # APROACHING STATE: run a kp speed and kp angle controller 
    if elevator_state == "APPROACHING":
        if (len(objs) != 0):
            pv_a = center[0]
            setpoint_a = rc.camera.get_width()
            error_a = setpoint_a - pv_a
            kp_a = 0.005
            angle = rc_utils.clamp(kp_a * error_a, -1, 1)

            pv_s = obj.bbox.area()
            setpoint_s = 4000
            error_s = setpoint_s - pv_s
            kp_s = 0.003
            speed = rc_utils.clamp(kp_s * error_s, 0, 1)
            last_angle = angle
            if pv_s == setpoint_s: # once we hit that target, go to the next state
                elevator_state = "WAITING"
        else:
            speed = MIN_SPEED
            angle = 0
    elif elevator_state == "WAITING": # Waiting right before the elevator
        # counter for a little bit of time to get on to the elevator
        if id == "Stop":
            speed = 0
            angle = 0
        elif id == "Go" and forward_counter < FORWARD_TIME:
            speed = MIN_SPEED
            angle = 0
            forward_counter += rc.get_delta_time()
        elif id == "Go" and forward_counter >= FORWARD_TIME:
            speed = 0
            angle = 0
            forward_counter = 0
            elevator_state = "AT_BOTTOM"
        else:
            speed = 0
            angle = 0
    elif elevator_state == "AT_BOTTOM": # Waiting for the sign to turn red to start the counter as it goes up
        update_contour()
        # initiate the counter once it turns red
        if contour_color == "RED":
            ascension_counter = 0
            elevator_state = "ASCENDING"
        speed = 0
        angle = 0
    elif elevator_state == "ASCENDING": # Going up on the elevator, counter based system for 7 seconds
        # 7 SECONDS IS BASED ON once it turns red, theres a 3 second buffer, 1 second to ascend, and it stays up for 10 seconds so we are extra safe here
        if ascension_counter > ASCENSION_TIME:
            ascension_counter = 0
            elevator_state = "EXITING"
        else:
            ascension_counter += rc.get_delta_time()
        speed = 0
        angle = 0
    elif elevator_state == "EXITING":
        # Exiting state will be wall following for a set time while its on the ramp - AR Tag at the end is very small so possibly risky
        if ramp_counter < RAMP_TIME:
            speed, angle = wall_follow()
            ramp_counter += rc.get_delta_time()
        else:
            elevator_state = "DONE"
            ramp_counter = 0
            speed = 0
            angle = 0
    return speed, angle #these need to be defined in this code - DONE

def hard_left():
    # simple counter to turn left for a good bit until we can clear that area right off the elevator
    global current_state
    if left_counter < LEFT_TIME:
        counter += rc.get_delta_time()
    else:
        counter = 0
        current_state = "LEFT_WALL_FOLLOW"
    angle = -1
    speed = MIN_SPEED
    return speed, angle
    

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
    if marker_id == 1 and marker_area >= MIN_AR_AREA: #TODO: we can change this value depending on tuning
        current_state = "ELEVATOR"
    #the "DONE" elevator state is selected after we've wall followed down the ramp for a long enough time, and
    #this means that we are no longer doing the elevator so we can switch to what's immediately after, left wall following
    elif elevator_state == "DONE":
        current_state = "LEFT_WALL_FOLLOW"
    #we only need to left wall follow to make sure we select the correct path in the area with all the lines after the elevator.
    #this means that once we see ar tag 5 right before the tunnel and the S bend, we can switch back to regular wall following
    elif marker_id == 5 and marker_area >= MIN_AR_AREA: #TODO: again, we will tune this number
        current_state = "WALL_FOLLOW"
    #LOGIC FOR WHAT TO DO DEPENDING ON WHAT STATE
    #these are lowkey pretty self explanatory, all these functions have built in controllers and data retrieval functions and
    #return what angle and speed the car should be at
    if current_state == "WALL_FOLLOW":
        speed, angle = wall_follow()
    elif current_state == "ELEVATOR":
        speed, angle = ride_elevator()
    elif current_state == "HARD_LEFT":
        speed, angle = hard_left()
    else:
        speed, angle = wall_follow_left()
    #sets speed and angle based on what was returned
    rc.drive.set_speed_angle(speed, angle)
    # dot matrix display
    display_matrix = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    sign = "-" if angle < 0 else "+"
    angle_str = "{:.2f}".format(angle) if sign == "+" else "{:.2f}".format(angle)[1:]
    pos_ones = angle_str[0]
    pos_tenths = angle_str[2]
    pos_hunds = angle_str[3]
    print(f"{angle_str}")

    insert_blank(display_matrix)
    insert_chars(display_matrix, sign)
    insert_blank(display_matrix)
    insert_nums(display_matrix, pos_ones)
    insert_blank(display_matrix)
    insert_chars(display_matrix, ".")
    insert_blank(display_matrix)
    insert_nums(display_matrix, pos_tenths)
    insert_blank(display_matrix)
    insert_nums(display_matrix, pos_hunds)
    
    rc.display.set_matrix(display_matrix)
    
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
