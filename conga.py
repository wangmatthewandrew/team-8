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
# TODO: change this stuff to match our file structure
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
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)
 # getting the object from the model
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
# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass

# Speed and angle controller based on the bounding box of the car in front
def update():
    global speed, angle, last_angle
    image = rc.camera.get_color_image()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, inference_size)
    run_inference(interpreter, rgb_image.tobytes())
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]
    center, id, obj = get_obj_and_type(image, inference_size, objs)

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
    else:
        speed = 0.75
        angle = -1 * last_angle
        
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
