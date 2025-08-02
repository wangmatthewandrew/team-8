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
def fix_angle(deg):
    return deg + 360 if deg < 0 else deg

def start():
    global speed, angle
    rc.drive.set_max_speed(1)
    speed = 1.0
    angle = 0.0
    rc.set_update_slow_time(0.5)

def update():
    global speed, angle, counter
#initialize stuff
    scan_data = rc.lidar.get_samples()
    counter += rc.get_delta_time()
    fov_span = 120
    fan_width = 30
    scan_limit = 100
    min_gap = 120
    step = 2

    chosen_heading = 0
    best_opening = 0
        #this part iterates on each potential heading from -75 to 75, for the left wall follow version we change this window to
    #make it biased towards the left. we basically prevent it from seeing openings to the right
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
        #try to find best opening
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
    # if counter <6 and counter >1:
    #     merged_angle = rc_utils.clamp(merged_angle-.7, -1, 1)
    #     print("correcting")
    # speed = 1.0 if best_opening > 220 else 0.6

    #proportional speed controller
    kp_s = 1
    speed = rc_utils.clamp(kp_s * (1 - abs(merged_angle)), 0.6, 1)
    print(f"speed: {speed}")
    rc.drive.set_speed_angle(speed, merged_angle)

def update_slow():
    pass

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
