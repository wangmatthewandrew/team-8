import sys
import math
import numpy as np

sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils
import cv2
import os
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

rc = racecar_core.create_racecar()



speed = 0.0
angle = 0.0
STATES = {1: "STOP", 2: "YIELD",  3: "GO_AROUND", 4: "NONE"}
current_state = STATES[4]
YIELD_SPEED = .65
SLOW_SPEED = .7
NORMAL_SPEED = .9
#threshold for the score of a detected object for it to be considered
SCORE_THRESH = 0.1
#how many types of objects we have
NUM_CLASSES = 3

go_around_sequence = False
go_around_counter = 0
#TODO: CHANGE THESE ############
default_path = 'conga_models' # location of model weights and labels
model_name = 'car_edgetpu.tflite'
label_name = 'car_labels.txt'
#find model and make interpereter object
model_path = default_path + "/" + model_name
label_path = default_path + "/" + label_name
print('Loading {} with {} labels.'.format(model_path, label_path))
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

go_around_counter = 0
go_around_sequence = False

#this function returns given angle but in positive form (but same location)
def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

def start():
    global speed, angle
    rc.drive.set_max_speed(1.0)
    speed = 1.0
    angle = 0.0
    rc.set_update_slow_time(0.5)
#this returns the most confident detected thing's center, id, and object
def get_obj_and_type(cv2_im, inference_size, objs):
    height, width, _ = cv2_im.shape
    max_score = 0
    correct_obj = None
    #scale the bbox
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    #iterate thru each obj and get the one with max score
    for obj in objs:
        if obj.score > max_score:
            max_score = obj.score
            correct_obj = obj
    #if there is an obj, get center using midpt formula nad get id
    if (correct_obj is not None):
        bbox = correct_obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        center = ((x0+x1)/2, (y0+y1)/2)
        id = correct_obj.id
    return center, id, correct_obj

def update():
    global speed, angle
    global go_around_counter
    global go_around_sequence
    scan_data = rc.lidar.get_samples()
    ### update the sign
    image = rc.camera.get_color_image()
    if image is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, inference_size)
        run_inference(interpreter, rgb_image.tobytes())
        objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]    
        if (len(objs) != 0):
            center, id, obj = get_obj_and_type(image, inference_size, objs)
    #depending on what the obj is, change the state
    if id == "Stop Sign":
        current_state = STATES[1]
    elif id == "Yield Sign":
        current_state = STATES[2]
    elif id == STATES[3]:
        current_state = STATES[3]
    else:
        current_state = STATES[4]
    fov_span = 120
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2
    chosen_heading = 0
    best_opening = 0
    #stop if its a stop sign
    if current_state == STATES[1]:
        speed = 0
    #if its yield, still wall follow but set slow
    elif current_state == STATES[2]:
        #this part iterates on each potential heading from -75 to 75, for the left wall follow version we change this window to
    #make it biased towards the left. we basically prevent it from seeing openings to the right
        for heading in range(-75, 75, step):
            start = heading - fan_width // 2
            end = heading + fan_width // 2

            samples = []
            #iterate thru angles in each potential opening
            for ang in range(start, end + 1):
                adjusted = fix_angle(ang)
                dist = rc_utils.get_lidar_average_distance(scan_data, adjusted)
                if dist is not None and dist > scan_limit:
                    samples.append(dist)

            if not samples or min(samples) < min_gap:
                continue
            #try to choose the bset opening
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

        merged_angle = rc_utils.clamp((chosen_heading / 70.0 + wall_adjust) / 2.0, -1.0, 1.0)
        #set speed slower
        speed =YIELD_SPEED
    elif current_state == STATES[3] or go_around_sequence == True:
        go_around_sequence = True
        if (go_around_counter <= 2):
            angle = 1
            counter += rc.get_detla_time()
        elif (go_around_counter <= 4):
            angle = -1
            counter += rc.get_delta_time()
        else:
            go_around_sequence = False
            counter = 0
        speed = 1
    elif current_state == STATES[4] and go_around_sequence == False:
        #this part iterates on each potential heading from -75 to 75, for the left wall follow version we change this window to
    #make it biased towards the left. we basically prevent it from seeing openings to the right
        for heading in range(-75, 75, step):
            start = heading - fan_width // 2
            end = heading + fan_width // 2

            samples = []
            #iterate thru angles in each potential opening
            for ang in range(start, end + 1):
                adjusted = fix_angle(ang)
                dist = rc_utils.get_lidar_average_distance(scan_data, adjusted)
                if dist is not None and dist > scan_limit:
                    samples.append(dist)

            if not samples or min(samples) < min_gap:
                continue
            #try to get best opening
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

        merged_angle = rc_utils.clamp((chosen_heading / 70.0 + wall_adjust) / 2.0, -1.0, 1.0)
        #bang bang speed controller
        speed = NORMAL_SPEED if best_opening > 220 else SLOW_SPEED

    rc.drive.set_speed_angle(speed, merged_angle)
    #print(f"Attitude: ")

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
