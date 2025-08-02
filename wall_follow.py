import sys
import math
import numpy as np

sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()
speed = 0.0
angle = 0.0
counter = 0

# [FUNCTION] Maps all angles to be between 0 and 360 (failsafe for if angles somehow bug out)
def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

# [FUNCTION] Initializes speed, angle, and printing time for debugging
def start():
    global speed, angle
    rc.drive.set_max_speed(1) # This sets the RACECAR speed to match our real-life speed in the sim
    speed = 1.0
    angle = 0.0
    rc.set_update_slow_time(0.5)

# [FUNCTION] Updates every frame, adjusts angle and speed, then sends to the plant (RACECAR)
def update():
    global speed, angle, counter

    # [PART 1] Localized path planner (sort of)
    # Initialize varianbles
    scan_data = rc.lidar.get_samples() # LIDAR scan
    fov_span = 120 
    fan_width = 30 # How far about each angle to check distance
    scan_limit = 100
    min_gap = 120 # Minimum safe distance 
    step = 2 # How far apart angles will be in the range of headings

    chosen_heading = 0 # Heading of car
    best_opening = 0 # How large opening is
    
    # Iterates on each potential heading from -75 to 75 (gap of 2), for the left wall follow version we change this window to
    # make it biased towards the left (change 75 to 20, leave -75). we basically prevent it from seeing openings to the right
    for heading in range(-75, 75, step):
        start = heading - fan_width // 2 # Adjusts start angle of scan
        end = heading + fan_width // 2 # Adjusts end angle of scan

        # Finds the averaged distances for every angle from -75 to 75
        samples = []
        for ang in range(start, end + 1): # Checks the distances for the range about an angle
            adjusted = fix_angle(ang) 
            dist = rc_utils.get_lidar_average_distance(scan_data, adjusted) 
            if dist is not None and dist > scan_limit: # Checks that the distance is far enough (to account for the reactive nature of the wall follower)
                samples.append(dist)

        if not samples or min(samples) < min_gap: # If samples is empty or the there are no gaps (distance from car) large enough, continue
            continue
            
        # Iterates through the available gaps to find the minimum (best) heading for the car
        candidate_clearance = min(samples)
        if candidate_clearance > best_opening:
            chosen_heading = heading
            best_opening = candidate_clearance

    # [PART 2] "Basic" reactive PID controller (its just P, :P)
    special_light = 60 # Angle of the PID controller (look ahead a little to account for reactive nature of controller)
    sample_window = 2 # Analogous to "fan_width"
    kp = 0.003

    # Lidar angles
    r_angle, r_dist = rc_utils.get_lidar_closest_point(scan_data, (0, 180)) # Finds closest point ahead
    l_angle, l_dist = rc_utils.get_lidar_closest_point(scan_data, (180, 360)) # Finds closest point behind
    r_shift = rc_utils.get_lidar_average_distance(scan_data, special_light, sample_window) # Finds average distance around the right angle
    l_shift = rc_utils.get_lidar_average_distance(scan_data, 360 - special_light, sample_window) # Finds average distance around the left angle

    r_component = math.sqrt(max(0, r_shift ** 2 - r_dist ** 2))
    l_component = math.sqrt(max(0, l_shift ** 2 - l_dist ** 2))

    # PID (its just P, :P)
    error = r_component - l_component 
    wall_adjust = rc_utils.clamp(error * kp, -1, 1)

    # Merge both the PID (really just P) part and the path planning part
    merged_angle = rc_utils.clamp((chosen_heading / 30.0 + wall_adjust) / 2.0, -1.0, 1.0)

    # Proportional speed controller
    kp_s = 1
    speed = rc_utils.clamp(kp_s * (1 - abs(merged_angle)), 0.6, 1)
    print(f"speed: {speed}")
    rc.drive.set_speed_angle(speed, merged_angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
