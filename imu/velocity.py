"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-summer-labs

File Name: lab_2b.py

Title: BYOA (Build Your Own AHRS)

Author: Bang-Bang (Team 8)

Purpose: The goal of this lab is to build and deploy a ROS node that can ingest
IMU data and return accurate velocity estimates (x', y', z') that can then
be used for autonomous navigation. It is recommended to review the equations of
motion and axes directions for the RACECAR Neo platform before starting. Template
code has been provided for the implementation of a Complementary Filter.

Expected Outcome: Subscribe to the /imu and /mag topics, and publish to the /velo
topic with accurate attitude estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3
import numpy as np
import math
import time

class VeloNode(Node):
    def __init__(self):
        super().__init__('velo_node')

        # Set up subscriber and publisher nodes
        self.subscription_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.subscription_mag = self.create_subscription(MagneticField, '/mag', self.mag_callback, 10)
        self.publisher_velo = self.create_publisher(Vector3, '/velo', 10) # output as [x', y', theta'] speeds

        self.prev_time = self.get_clock().now() # initialize time checkpoint

        # Set up attitude params
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha = 0.95

        # Set up velocity params
        self.x_velo = 0.0
        self.y_velo = 0.0
        self.z_velo = 0.0
        self.mag = None

        # Set up accel bias params
        self.x_bias = -0.05 
        self.y_bias = 0 
        self.z_bias = 0 

        # Kalman Filter params
        self.x_pred = None # stored predicted velocity
        self.p_pred = None # stored predicted covariance
        self.x = np.zeros((3))  # [vx, vy, vz]
        self.P = np.eye(3) * 1  # covariance
        self.Q = np.eye(3) * 0.2 # process noise [vx, vy, vz]
        self.R = np.eye(3) * 0.2 # measurement noise (accelerometer)

    # [FUNCTION] Called when magnetometer topic receives an update
    def mag_callback(self, data):
        #Assign self.mag to the magnetometer data points
        self.mag = (data.magnetic_field.x, data.magnetic_field.y, data.magnetic_field.z)

    # [FUNCTION] Called when new IMU data is received, velocity calc completed here as well
    def imu_callback(self, data):
        #Grab linear acceleration and angular velocity values from subscribed data points
        accel = data.linear_acceleration
        gyro = data.angular_velocity

        #Calculate time delta
        now = self.get_clock().now() # Current ROS time
        dt = now - self.prev_time # Time delta
        dt = dt.nanoseconds / 1e9
        self.prev_time = now # refresh checkpoint

        # Derive tilt angles from accelerometer
        accel_roll = math.atan2(accel.y, math.sqrt(accel.x ** 2 + accel.z ** 2)) # theta_x
        accel_pitch = math.atan2(-accel.x, math.sqrt(accel.y ** 2 + accel.z ** 2)) # theta_y

        # Integrate gyroscope to get attitude angles
        gyro_roll = self.roll + gyro.x * dt # theta_xt
        gyro_pitch = self.pitch + gyro.y * dt # theta_yt
        gyro_yaw = self.yaw + gyro.z * dt # theta_zt

        # Compute yaw angle from magnetometer
        if self.mag:
            mx, my, mz = self.mag
            print(f"Mag norm (~50 uT): {math.sqrt(mx**2 + my**2 + mz**2) * 1e6}") # used for checking magnetic disturbances/offsets
            bx = (mx * math.cos(accel_roll)) + (mz * math.sin(accel_pitch))
            by = (mx * math.sin(accel_roll) * math.sin(accel_pitch)) + (my * math.cos(accel_roll)) - (mz * math.sin(accel_roll) * math.cos(accel_pitch))
            mag_accel_yaw = math.atan2(by, bx) + math.pi
        else:
            mag_accel_yaw = self.yaw
        
        # Fuse gyro, mag, and accel derivations in complementary filter
        self.roll = (self.alpha * gyro_roll) + ((1 - self.alpha) * accel_roll)
        self.pitch = (self.alpha * gyro_pitch) + ((1 - self.alpha) * accel_pitch)
        self.yaw = (self.alpha * gyro_yaw) + ((1 - self.alpha) * mag_accel_yaw)

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
        true_accel = R_global @ np.array([accel.x, accel.y, accel.z]) - grav
        
        # High Pass Filter for acceleration
        x_accel = true_accel[0] + self.x_bias 
        y_accel = true_accel[1] + self.y_bias
        z_accel = true_accel[2] + self.z_bias
        if abs(x_accel) < 1e-1:
            x_accel = 0
        if abs(y_accel) < 1e-1:
            y_accel = 0
        if abs(z_accel) < 1e-1:
            z_accel = 0

        # True Acceleration
        true_accel = np.array([x_accel, y_accel, z_accel])
        
        # Derive velocities from accelerometer (3D Kalman Filter)
        self.x_pred = self.x + true_accel * dt
        F_jac = np.eye(3)
        self.P_pred = F_jac @ self.P @ F_jac.T + self.Q
        y = (true_accel - ((self.x_pred - self.x) / dt)).flatten()
        H_jac = np.eye(3)
        self.S = H_jac @ self.P_pred @ H_jac.T + self.R
        K_gain = self.P_pred @ H_jac.T @ np.linalg.inv(self.S)

        # Update velocity and covariance
        self.x = (self.x_pred + K_gain @ y).flatten()
        self.P = (np.eye(3) - K_gain @ H_jac) @ self.P_pred

        # Store accelerometer values
        self.x_velo = self.x[0]
        self.y_velo = self.x[1]
        self.z_velo = self.x[2]

        # Print results for sanity checking
        print(f"====== Velocity Integration Results ======")
        print(f"Speed || Freq = {round(1/dt,0)} || dt (ms) = {round(dt*1e3, 2)}")
        print()
        print(f"Gravity")
        print(f"Attitude: {[ self.roll * 180/math.pi, self.pitch * 180/math.pi, self.yaw * 180/math.pi]}")
        print(f"Gravity Vector: {grav}")
        print(f"True Accel: {true_accel}")
        print()
        print(f"Results")
        print(f"X Velocity (m/s): {self.x_velo}")
        print(f"Y Velocity (m/s): {self.y_velo}")
        print(f"Z Velocity (m/s): {self.z_velo}")
        print(f"X Accel: {accel.x}, Y Accel; {accel.y} Z Accel: {accel.z}")
        print("\n")
        
        #Publish to velocity topic
        velo = Vector3()
        velo.x = self.x_velo
        velo.y = self.y_velo
        velo.z = self.z_velo
        self.publisher_velo.publish(velo)
    
def main():
    rclpy.init(args=None)
    node = VeloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
