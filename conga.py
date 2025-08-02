"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: conga.py << [Modify with your own file name!]

Title: Conga Line (Trial 3B) << [Modify with your own title]

Author: Bang-Bang (Team 8) << [Write your name or team name here]

Purpose: To follow a car autonomously

Expected Outcome: Using a trained neural network on the TPU of the car, detect and autonomously
follow a wall following car to the end of the Universal Track.
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
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

# Define paths to model and label directories
# Changed this stuff to match our file structure
default_path = 'models' # location of model weights and labels
model_name = 'efficientdet0_edgetpu.tflite'
label_name = 'labels.txt'

model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name

# Define thresholds and number of classes to output
SCORE_THRESH = 0.1
NUM_CLASSES = 3
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
# Load the model onto the car
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

# [FUNCTION] Getting the object, center of object, and id from the model
def get_obj_and_type(cv2_im, inference_size, objs):
    height, width, _ = cv2_im.shape
    max_score = 0
    correct_obj = None
    scale_x, scale_y = width / inference_size[0], height / inference_size[1] # Get scaling values to resize
    for obj in objs: # Find the object that has the highest inference score, and set that to be correct_obj
        if obj.score > max_score:
            max_score = obj.score
            correct_obj = obj
    bbox = correct_obj.bbox.scale(scale_x, scale_y) # Scale the bounding box for correct_obj
    x0, y0 = int(bbox.xmin), int(bbox.ymin) # Get x_min and y_min
    x1, y1 = int(bbox.xmax), int(bbox.ymax) # Get x_max and y_max
    center = ((x0 + x1) / 2, (y0 + y1) / 2) # Find the center of the bounding box (object)
    id = correct_obj.id
    return center, id, correct_obj
 
# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass

# Speed and angle controller based on the bounding box of the car in front
def update():
    global speed, angle, last_angle
    image = rc.camera.get_color_image()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR raw image to RGB (for the NN)
    rgb_image = cv2.resize(rgb_image, inference_size)
    run_inference(interpreter, rgb_image.tobytes())
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES] # Get objects
    center, id, obj = get_obj_and_type(image, inference_size, objs) # Get center, id, and object of best object

    if (len(objs) != 0):
        # basic proportional angle controller
        pv_a = center[0]
        setpoint_a = rc.camera.get_width()
        error_a = setpoint_a - pv_a
        kp_a = 0.005
        angle = rc_utils.clamp(kp_a * error_a, -1, 1)

        # Basic proportional speed controller
        pv_s = obj.bbox.area()
        setpoint_s = 4000
        error_s = setpoint_s - pv_s
        kp_s = 0.005
        speed = rc_utils.clamp(kp_s * error_s, 0, 1)
        last_angle = angle
    else: # Speed up and reverse the other way (likely lost vision of car)
        speed = 0.75
        angle = -1 * last_angle
        
    rc.drive.set_speed_angle(speed, angle) # Set speed and angle

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
