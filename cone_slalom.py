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

ORANGE = ((5, 130, 184), (38, 255, 255))
# ORANGE = ((1, 30 210), (30, 255, 255))
# GREEN = ((30, 35, 100), (55, 255, 255))
GREEN = ((31, 101, 33), (75, 255, 255))

STATES = {1: "ORANGE", 2: "GREEN", 3: "FIND_ORANGE", 4: "FIND_GREEN"}
current_state = STATES[1]
previous_state = "NONE"

queue = []

previous = ""

########################################################################################
# Functions
########################################################################################

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
        contours_orange = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])
        contours_green = rc_utils.find_contours(image, GREEN[0], GREEN[1])
        # print("skib 1")
        for contour in contours_orange:
            if cv.contourArea(contour) > MIN_CONTOUR_AREA:
                contour_list.append((contour, "ORANGE"))
                print("seeing orange")
            #print("looping 1")
        for contour in contours_green:
            if cv.contourArea(contour) > MIN_CONTOUR_AREA:
                contour_list.append((contour, "GREEN"))
            #print("looping 2")
    
        if contour_list:
            max_contour = max(contour_list, key = lambda x: cv.contourArea(x[0]))
            color = max_contour[1]
            max_contour = max_contour[0]
            contour_area = cv.contourArea(max_contour)
            contour_center = rc_utils.get_contour_center(max_contour)
            cv.drawContours(image, [max_contour], -1, (0, 0, 0), 3)
        if (color == "ORANGE"):
            current_state = "ORANGE"
            previous_state = "ORANGE"
        elif (color == "GREEN"):
            current_state = "GREEN"
            previous_state = "GREEN"
        else:
            cone_num += 1
            if (previous_state == "ORANGE"):
                current_state = "FIND_GREEN"
            else:
                current_state = "FIND_ORANGE"
                # previous_state = "FIND_BLUE"
       # print("finished all of the above")
        contour_list.clear()
    else:
        print("that silly goober image is empty")
    


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    speed = 1
    update_contour()
    # queue.clear() # Remove 'pass' and write your source code for the start() function here

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
    driving_speed = 0.875
    print("contour area: " , contour_area , "| contour center: " , contour_center)
    # print(rc_utils.get_lidar_closest_point(scan, (30, 180)))
     # if it sees the cone, turn away until it doenst see it anymore
    if (current_state == "ORANGE"):
        print("turn left")
        speed = driving_speed
        update_contour()
        if (contour_center is not None):
            present_value = contour_center[1]
        else:
            present_value = 0
        setpoint = rc.camera.get_width() * 0.99
        error = setpoint - present_value
        kp = -0.005
        angle = rc_utils.clamp(kp * error, -1, 1)
    elif (current_state == "GREEN"):
        print("turn right")
        speed = driving_speed
        update_contour()
        if (contour_center is not None):
            present_value = contour_center[1]
        else:
            present_value = 0
        setpoint = rc.camera.get_width() * 0.05
        error = setpoint - present_value
        kp = -0.004
        angle = rc_utils.clamp(kp * error, -1, 1)
    # after it turns away, if it's still next to the cone, keep going straight (with a small offset)
    # if its past the previous cone, start turning back to find the next code
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
