"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: lab_2a.py

Title: BYOA (Build Your Own AHRS)

Author: Bang-Bang (Team 8)

Purpose: The goal of this lab is to build and deploy a ROS node that can ingest
IMU data and return accurate attitude estimates (roll, pitch, yaw) that can then
be used for autonomous navigation. It is recommended to review the equations of
motion and axes directions for the RACECAR Neo platform before starting. Template
code has been provided for the implementation of a Complementary Filter.

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /attitude
topic with accurate attitude estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3
import numpy as np
import math
import time

class AttitudeCompNode(Node):
    def __init__(self):
        super().__init__('attitude_node')

        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)
        self.publisher_attitude = self.create_publisher(Vector3, '/attitude', 10) # output as [roll, pitch, yaw] angles

        self.prev_time = self.get_clock().now() # initialize time checkpoint
        self.alpha = 0.95 # TODO: Determine an alpha value that works with the complementary filter

        # set up attitude params
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.mag = None
    
    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # TODO: Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z)

    # [FUNCTION] Called when new IMU data is received, attitude calc completed here as well
    def imu_callback(self, data):
        # TODO: Grab linear acceleration and angular velocity values from subscribed data points
        accel = data.linear_acceleration
        gyro = data.angular_velocity

        # TODO: Calculate time delta
        now = self.get_clock().now() # Current ROS time
        dt = now - self.prev_time  # Time delta
        dt = dt.nanoseconds / 1e9
        self.prev_time = now # refresh checkpoint
    
        # TODO: Derive tilt angles from accelerometer
        accel_roll = math.atan2(accel.y, math.sqrt(accel.x ** 2 + accel.z ** 2)) # theta_x
        accel_pitch = math.atan2(-accel.x, math.sqrt(accel.y ** 2 + accel.z ** 2)) # theta_y - seems correct?

        # TODO: Integrate gyroscope to get attitude angles
        gyro_roll = self.roll + gyro.x * dt # theta_xt
        gyro_pitch = self.pitch + gyro.y * dt # theta_yt
        gyro_yaw = self.yaw + gyro.z * dt # theta_zt

        # TODO: Compute yaw angle from magnetometer
        if self.mag:
            mx, my, mz = self.mag
            print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            bx = (mx * math.cos(accel_roll)) + (mz * math.sin(accel_pitch))
            by = (mx * math.sin(accel_roll) * math.sin(accel_pitch)) + (my * math.cos(accel_roll)) - (mz * math.sin(accel_roll) * math.cos(accel_pitch))
            mag_accel_yaw = math.atan2(by, bx) + math.pi
        else:
            mag_accel_yaw = self.yaw
        
        # TODO: Fuse gyro, mag, and accel derivations in complementary filter
        self.roll = (self.alpha * gyro_roll) + ((1 - self.alpha) * accel_roll)
        self.pitch = (self.alpha * gyro_pitch) + ((1 - self.alpha) * accel_pitch)
        self.yaw = (self.alpha * gyro_yaw) + ((1 - self.alpha) * mag_accel_yaw)

        # Print results for sanity checking
        print(f"====== Complementary Filter Results ======")
        print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
        print(f"Accel + Mag Derivation")
        print(f"Roll (deg): {accel_roll * 180/math.pi % 360}")
        print(f"Pitch (deg): {accel_pitch * 180/math.pi % 360}")
        print(f"Yaw (deg): {mag_accel_yaw * 180/math.pi % 360}")
        print()
        print(f"Gyro Derivation")
        print(f"Roll (deg): {gyro_roll * 180/math.pi % 360}")
        print(f"Pitch (deg): {gyro_pitch * 180/math.pi % 360}")
        print(f"Yaw (deg): {gyro_yaw * 180/math.pi % 360}")
        print()
        print(f"Fused Results")
        print(f"Roll (deg): {self.roll * 180/math.pi % 360}")
        print(f"Pitch (deg): {self.pitch * 180/math.pi % 360}")
        print(f"Yaw (deg): {self.yaw * 180/math.pi % 360}")
        print("\n")
        
        # TODO: Publish to attitude topic (convert to degrees)
        attitude = Vector3()
        attitude.x = self.roll * 180 / math.pi % 360
        attitude.y = self.pitch * 180 / math.pi % 360
        attitude.z = self.yaw * 180 / math.pi % 360
        self.publisher_attitude.publish(attitude)
    
def main():
    rclpy.init(args=None)
    node = AttitudeCompNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
