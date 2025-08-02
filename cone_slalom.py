########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import time

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
contour_list = []
counter = 0.0
cone_num = 1

MIN_CONTOUR_AREA = 1500

ORANGE = ((5, 130, 184), (38, 255, 255)) # HSV Range for Orange
GREEN = ((31, 101, 33), (75, 255, 255)) # HSV Range for Green

STATES = {1: "ORANGE", 2: "GREEN", 3: "FIND_ORANGE", 4: "FIND_GREEN"}
current_state = STATES[1]
previous_state = "NONE"

queue = []

previous = ""

########################################################################################
# Functions
########################################################################################

# [FUNCTION] Updates contour (to detect the cone)
def update_contour():
    global contour_center
    global contour_area
    global contour_list
    global current_state
    global previous_state
    global cone_num
    
    color = ""

    image = rc.camera.get_color_image()
    if (image is not None):
        contours_orange = rc_utils.find_contours(image, ORANGE[0], ORANGE[1]) # Finds the orange contours
        contours_green = rc_utils.find_contours(image, GREEN[0], GREEN[1]) # Finds the green contours

        # Finds orange contours
        for contour in contours_orange:
            if cv.contourArea(contour) > MIN_CONTOUR_AREA:
                contour_list.append((contour, "ORANGE")) # Appends as (contour, color) tuple
                print("seeing orange")

        # Finds green contours
        for contour in contours_green:
            if cv.contourArea(contour) > MIN_CONTOUR_AREA:
                contour_list.append((contour, "GREEN")) # Appends as (contour, color) tuple
                print("seeing green")

        # Finds largest contour
        if contour_list:
            max_contour = max(contour_list, key = lambda x: cv.contourArea(x[0])) # Finds largest size contour
            color = max_contour[1]
            max_contour = max_contour[0]
            contour_area = cv.contourArea(max_contour)
            contour_center = rc_utils.get_contour_center(max_contour) # Gets the center of the largest contour
            cv.drawContours(image, [max_contour], -1, (0, 0, 0), 3) # Draws contours for sanity checking
            
        if (color == "ORANGE"):
            current_state = "ORANGE"
            previous_state = "ORANGE"
        elif (color == "GREEN"):
            current_state = "GREEN"
            previous_state = "GREEN"
        else: # If we passed a cone we switch state to finding states
            cone_num += 1
            if (previous_state == "ORANGE"): # After passing orange, search for green
                current_state = "FIND_GREEN"
            else: # After passing green, search for orange
                current_state = "FIND_ORANGE"

        # Clears list of contours to start again
        contour_list.clear()
    else:
        print("that silly goober image is empty") # truly a bruh moment
    
# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    speed = 1
    update_contour()

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global queue
    global current_state
    global previous_state
    global contour_center
    global speed
    global angle
    
    scan = rc.lidar.get_samples()
    # print(current_state)
    driving_speed = 0.875 # This speed is off (we had 1 speed upgrade at the time)
    print("contour area: " , contour_area , "| contour center: " , contour_center) # Print contour for sanity checking

     # If it sees the cone, turn away until it doenst see it anymore (state machine)
    if (current_state == "ORANGE"):
        print("turn left")
        speed = driving_speed
        update_contour()
        if (contour_center is not None):
            present_value = contour_center[1] # The x value of contour center (center of the cone)
        else:
            present_value = 0
        # Basic proportional angle control (the setpoint is very far right)
        setpoint = rc.camera.get_width() * 0.99
        error = setpoint - present_value
        kp = -0.005
        angle = rc_utils.clamp(kp * error, -1, 1)
    
    elif (current_state == "GREEN"):
        print("turn right")
        speed = driving_speed
        update_contour()
        if (contour_center is not None):
            present_value = contour_center[1] # The x value of contour center (center of the cone)
        else:
            present_value = 0
        # Basic proportional angle control (the setpoint is very far right)
        setpoint = rc.camera.get_width() * 0.05
        error = setpoint - present_value
        kp = -0.004 # This kp is different because mechanical issues
        angle = rc_utils.clamp(kp * error, -1, 1)
        
    # After it turns away, if it's still next to the cone, keep going straight (with a small offset)
    # If its past the previous cone, start turning back to find the next code
    elif (current_state == "FIND_ORANGE"):
        closest_cone = 0
        if (len(scan) >= 1):
            closest_cone,  _ = rc_utils.get_lidar_closest_point(scan, (180, 10))
        if (closest_cone > 280):
            print("go around green")
            speed = driving_speed
            angle = -0.1
        else:
            print("find orange")
            speed = driving_speed
            angle = -1
            update_contour()
    
    elif (current_state == "FIND_GREEN"):
        closest_cone = 0
        if (len(scan) >= 1):
            closest_cone, _ = rc_utils.get_lidar_closest_point(scan, (10, 180))
        if (closest_cone < 80):
            print("closest object at: " , closest_cone)
            print("go around orange")
            speed = driving_speed
            angle = 0.2
        else:
            print("find green")
            speed = driving_speed
            angle = 1
            update_contour()
    else:
        print("none")
        speed = 0
        angle = 0
        update_contour()
    rc.drive.set_speed_angle(speed, angle)
    # Remove 'pass' and write your source code for the update() function here

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
