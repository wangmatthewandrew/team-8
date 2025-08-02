"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: lab_2a.py

Title: BYOA (Build Your Own AHRS)

Author: Bang-Bang (Team 8)

Purpose: The goal of this lab is to build and deploy a ROS node that can ingest
IMU data and return accurate pose estimates (x, y, theta) that can then
be used for autonomous navigation. It is recommended to review the equations of
motion and axes directions for the RACECAR Neo platform before starting. Template
code has been provided for the implementation of a Complementary Filter.

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /pose
topic with accurate pose estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3
import numpy as np
import math
import time

class PoseEstiNode(Node):
    def __init__(self):
        super().__init__('pose_node')

        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)
        self.publisher_pose_estimate = self.create_publisher(Vector3, '/pose', 10) # output as [x, y, theta] pose

        self.prev_time = self.get_clock().now() # initialize time checkpoint

        # Set up attitude params
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha = 0.95

        # Set up accel bias params
        self.x_bias = -0.05
        self.y_bias = 0

        # Set up pose estimate params
        self.angle = 0.0
        self.flat_accel = 0.0
        self.mag = None

        # Set up Kalman Filter params
        self.x_pred = None # stored predicted pose
        self.p_pred = None # stored predicted covariance
        self.x = np.zeros((3, 1))  # [x, y, v]
        self.P = np.eye(3) * 1  # covariance
        self.Q = np.diag([0.5, 0.5, 0.2])  # process noise [x, y, v]
        self.R = np.array([[1]])  # measurement noise (accelerometer)
    
    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        # TODO: Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z)

    # [FUNCTION] Called when new IMU data is received, attitude calc completed here as well
    def imu_callback(self, data):
        # TODO: Grab linear acceleration and angular velocity values from subscribed data points
        accel = data.linear_acceleration
        gyro = data.angular_velocity

        # Calculate time delta
        now = self.get_clock().now() # Current ROS time
        dt = now - self.prev_time # Time delta
        dt = dt.nanoseconds / 1e9
        self.prev_time = now # refresh checkpoint
    
        # Derive tilt angles from accelerometer
        accel_roll = math.atan2(accel.y, math.sqrt(accel.x ** 2 + accel.z ** 2)) # theta_x
        accel_pitch = math.atan2(-accel.x, math.sqrt(accel.y ** 2 + accel.z ** 2)) # theta_y - seems correct?

        # Integrate gyroscope to get attitude yaw 
        gyro_roll = self.roll + gyro.x * dt # theta_xt
        gyro_pitch = self.pitch + gyro.y * dt # theta_yt
        gyro_yaw = self.yaw + gyro.z * dt # theta_zt

        # Compute yaw angle from magnetometer
        if self.mag:
            mx, my, mz = self.mag
            print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            bx = (my * math.cos(accel_pitch)) + (mz * math.sin(accel_pitch))
            by = (mx * math.sin(accel_roll) * math.sin(accel_pitch)) + (my * math.cos(accel_roll)) - (mz * math.sin(accel_roll) * math.cos(accel_pitch))
            mag_accel_yaw = math.atan2(by, bx) + math.pi
        else:
            mag_accel_yaw = self.yaw
        
        # Fuse the angle yaw derivations in complementary filter
        self.roll = (self.alpha * gyro_roll) + ((1 - self.alpha) * accel_roll)
        self.pitch = (self.alpha * gyro_pitch) + ((1 - self.alpha) * accel_pitch)
        self.yaw = (self.alpha * gyro_yaw) + ((1 - self.alpha) * mag_accel_yaw)
        self.angle = self.yaw

        # Finding gravity orientation
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(self.roll), -math.sin(self.roll)],
            [0, math.sin(self.roll), math.cos(self.roll)],
        ])
        R_y = np.array([
            [math.cos(self.pitch), 0, math.sin(self.pitch)],
            [0, 1, 0],
            [-math.sin(self.pitch), 0, math.cos(self.pitch)],
        ])
        R_z = np.array([
            [math.cos(self.yaw), -math.sin(self.yaw), 0],
            [math.sin(self.yaw), math.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        R_global = (R_z @ -R_y @ R_x).T # Rotates the IMU measurements to match the world frame
        grav = R_global @ np.array([0, 0, -9.89]) # 9.89 is gravity because the IMU is a bit inaccurate

        # Take the gravity vector away after rotating car measurements
        true_accel = (R_global @ np.array([accel.x, accel.y, accel.z])) - grav

        # High Pass Filter & Heading calculation
        x_accel = true_accel[0] + self.x_bias # 0.28
        y_accel = true_accel[1] + self.y_bias # 0.1
        self.flat_accel = math.sqrt(x_accel ** 2 + y_accel ** 2)
        heading = np.array([math.cos(self.angle), math.sin(self.angle)])
        accel = np.array([x_accel, y_accel])
        self.flat_accel = np.dot(accel, heading)
        if abs(self.flat_accel) < 5e-1:
            self.flat_accel = 0

        # 3D Extended Kalman Filter for Position and Velocity magnitude
        self.x_pos, self.y_pos, self.velo = self.x.flatten()
        # High Pass 2: Electric Boogaloo
        if abs(self.velo) < 3e-1:
            self.velo = 0
        self.x_pred = np.array([
            [self.x_pos + self.velo * math.cos(self.angle) * dt],
            [self.y_pos + self.velo * math.sin(self.angle) * dt],
            [self.velo + self.flat_accel * dt],
        ])
        F_jac = np.array([
            [1, 0, math.cos(self.angle) * dt],
            [0, 1, math.sin(self.angle) * dt],
            [0, 0, 1]
        ])
        self.P_pred = F_jac @ self.P @ F_jac.T + self.Q
        z = np.array([self.flat_accel])
        y = z - np.array([(self.x_pred[2] - self.x[2]) / dt])
        H_jac = np.array([[0, 0, 1 / dt]])
        self.S = H_jac @ self.P_pred @ H_jac.T + self.R
        K_gain = self.P_pred @ H_jac.T @ np.linalg.inv(self.S)

        # Update Pose & Covariance
        self.x = (self.x_pred + (K_gain @ y)).flatten() 
        self.P = (np.eye(3) - K_gain @ H_jac) @ self.P_pred

        # Update Positions & Velocity
        self.x_pos = self.x[0]
        self.y_pos = self.x[1]
        self.velo = self.x[2]

        # Print results for sanity checking
        print(f"====== Complementary/Kalman Filter Results ======")
        print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
        print(f"Accel + Mag Derivation")
        print(f"Roll (deg): {accel_roll * 180/math.pi}")
        print(f"Pitch (deg): {accel_pitch * 180/math.pi}")
        print(f"Yaw (deg): {mag_accel_yaw * 180/math.pi}")
        print()
        print(f"Gyro Derivation")
        print(f"Yaw (deg): {gyro_yaw * 180/math.pi}")
        print()
        print(f"Acceleration Raw")
        print(f"Gravity: {grav}")
        print(f"True Accel: {true_accel}")
        print(f"Flat Accel: {self.flat_accel}")
        print(f"Velo: {self.velo}")
        print()
        print(f"Pose Estimate")
        print(f"X Position (m): {self.x_pos}")
        print(f"Y Position (m): {self.y_pos}")
        print(f"Yaw/Angle (deg): {self.angle * 180/math.pi}")
        print("\n")
        
        # Publish to pose estimate topic
        pose = Vector3()
        pose.x = self.x_pos
        pose.y = self.y_pos
        pose.z = self.angle * 180 / math.pi
        self.publisher_pose_estimate.publish(pose)
    
def main():
    rclpy.init(args=None)
    node = PoseEstiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
