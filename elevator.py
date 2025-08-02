# STATES:
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

sys.path.insert(0, '../library')
import racecar_core
import cv2
import os
import time
import racecar_utils as rc_utils

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

STATES = {1: "APPROACHING", 2: "WAITING", 3: "AT_BOTTOM", 4: "ASCENDING", 5: "EXITING", 6: "DONE"}
current_state = "APPROACHING"

GREEN = ((30, 50, 50), (80, 255, 255))
RED = ((165, 50, 50), (20, 255, 255))
MIN_CONTOUR_AREA = 1000

FORWARD_TIME = 2
ASCENSION_TIME = 6.5

contour_color = ""
ascension_counter = 0
forward_counter = 0

# Define paths to model and label directories
# TODO: change this stuff to match our file structure
default_path = 'models' # location of model weights and labels
model_name = 'elevator_edgetpu.tflite'
label_name = 'elevator_labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 2
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0
angle = 0
last_angle = 0

########################################################################################
# Functions
########################################################################################

# Loading the model
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

# Using the model and returning center, id, and the obj object
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

# update contour just for the AT_BOTTOM state
def update_contour():
    global contour_point
    global contour_area
    global contour_color

    contour = None

    image = rc.camera.get_color_image()

    if image is None:
        contour_point = None
        contour_area = 0
    else:
        # Green and red values for the Go or Stop state
        green_contours = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        red_contours = rc_utils.find_contours(image, RED[0], RED[1])

        green_line = rc_utils.get_largest_contour(green_contours, MIN_CONTOUR_AREA)
        red_line = rc_utils.get_largest_contour(red_contours, MIN_CONTOUR_AREA)

        if red_line is not None:
            contour = red_line
            contour_color = "RED"
        elif green_line is not None:
            contour = green_line
            contour_color = "GREEN"

        if contour is not None:
            contour_point = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_point)

        # Display the image to the screen
    rc.display.show_color_image(image)

def start():
    pass

def update():
    global speed, angle, last_angle, ascension_counter, forward_counter
    # Getting all the images and the objects from the model
    image = rc.camera.get_color_image()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, inference_size)
    run_inference(interpreter, rgb_image.tobytes())
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]
    center, id, obj = get_obj_and_type(image, inference_size, objs)

    # APROACHING STATE: run a kp speed and kp angle controller 
    if current_state == "APPROACHING":
        if (len(objs) != 0):
            pv_a = center[0]
            setpoint_a = rc.camera.get_width()
            error_a = setpoint_a - pv_a
            kp_a = 0.005
            angle = rc_utils.clamp(kp_a * error_a, -1, 1)

            pv_s = obj.bbox.area()
            setpoint_s = 4000
            error_s = setpoint_s - pv_s
            kp_s = 0.005
            speed = rc_utils.clamp(kp_s * error_s, 0, 1)
            last_angle = angle

            if pv_s == setpoint_s
              current_state = "WAITING"
        else:
            speed = 0.75
            angle = -1 * last_angle
    elif current_state == "WAITING": # Waiting right before the elevator
        if id == 0:
            speed = 0
            angle = 0
        elif id == 1 and forward_counter < FORWARD_TIME:
            speed = 0.2
            angle = 0
        elif id == 1 and forward_counter >= FORWARD_TIME:
            speed = 0
            angle = 0
            forward_counter = 0
            current_state = "AT_BOTTOM"
        else:
            speed = 0
            angle = 0
    elif current_state == "AT_BOTTOM": # Waiting for the sign to turn red to start the counter as it goes up
        update_contour()
        if contour_color == "RED":
            ascension_counter = 0
            current_state = "ASCENDING"
        speed = 0
        angle = 0
    elif current_state == "ASCENDING": # Going up on the elevator, counter based system for 7 seconds
        if ascension_counter > ASCENSION_TIME:
            ascension_counter = 0
            current_state = "EXITING"
        else:
            counter += rc.get_delta_time()
        speed = 0
        angle = 0
    elif current_state == "EXITING":
        # Exiting state will be wall following, to be implemented in grand prix full code with full state machine
        speed = 1
        angle = 0
    else:
        # Set current state to global left wall following state
        pass
        
        
    rc.drive.set_speed_angle(speed, angle)

def update_slow():
    pass

# DO NOT MODIFY

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
