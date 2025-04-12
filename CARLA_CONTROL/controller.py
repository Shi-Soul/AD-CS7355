""" This module contains controllers to perform lateral and longitudinal control. """

import carla
import math
import numpy as np

from collections import deque
from utils import get_speed


class CUtils(object):
    def __init__(self):
        self.parameters = {}

    def set_param(self, param_name, value):
        self.parameters[param_name] = value

    def get_param(self, param_name, default_value=None):
        return self.parameters.get(param_name, default_value)


class Controller:
    """
    Controller combines both longitudinal and lateral controllers to manage the vehicle's movement in CARLA.
    """

    def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Initializes the Controller with maximum values for throttle, brake, and steering, along with the vehicle object.

        :param vehicle: The vehicle object to control.
        :param max_throttle: Maximum throttle value.
        :param max_brake: Maximum brake value.
        :param max_steering: Maximum steering angle.
        """
        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()  # Get the world object from the vehicle
        self.past_steering = (
            self._vehicle.get_control().steer
        )  # Store the current steering value for future reference

        self._lon_controller = CustomLongitudinalController(
            self._vehicle
        )  # Longitudinal controller instance
        self._lat_controller = CustomLateralController(
            self._vehicle
        )  # Lateral controller instance

    def run_step(self, target_speed, waypoints):
        """
        Executes one step of control using both the longitudinal and lateral controllers.

        :param target_speed: The desired vehicle speed in km/h.
        :param waypoints: The waypoints to follow, each waypoint is a carla.Waypoint object.
        :return: A carla.VehicleControl object containing the control commands for the vehicle.
        """
        # breakpoint()
        acceleration = self._lon_controller.run_step(
            target_speed
        )  # Get acceleration from longitudinal controller
        current_steering = self._lat_controller.run_step(
            waypoints
        )  # Get steering angle from lateral controller

        control = carla.VehicleControl()  # Initialize the vehicle control command
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering adjustments to ensure smooth changes and adherence to max steering values
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = (
            steering  # Update the past_steering value for the next control cycle
        )

        return control  # Return the control command for the vehicle


class CustomLongitudinalController:
    def __init__(self, vehicle, Kp=0.5, Ki=0.01, Kd=0.1, dt=0.1):
        self._vehicle = vehicle
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.dt = dt  # 控制周期（秒）
        self.integral = 0
        self.prev_error = 0

    def run_step(self, target_speed):
        current_speed = get_speed(self._vehicle)  # 当前速度（km/h）
        error = target_speed - current_speed  # 速度误差
        # print(f"Speed: error {error:.2f} km/h \t| {current_speed:.2f} km/h \t| {target_speed:.2f} km/h")  # Print current speed with formatting

        # PID计算
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        acceleration = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        # 限制输出范围 [-1, 1]
        return np.clip(acceleration, -1.0, 1.0)


class CustomLateralController:
    def __init__(self, vehicle, k_gain=5, k_soft=0.1, max_steer=0.8):
        self._vehicle = vehicle
        self.k_gain = k_gain  # 前轮转向增益
        self.k_soft = k_soft  # 软化系数（防止零速时奇异值）
        self.max_steer = max_steer  # 最大转向角限制

    def run_step(self, waypoints):
        if not waypoints:
            return 0.0

        # 获取车辆当前状态
        current_speed = get_speed(self._vehicle) / 3.6  # 转为m/s
        vehicle_transform = self._vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)
        front_axle = vehicle_location + carla.Location(
            x=np.cos(vehicle_yaw) * 3.0,  # 假设前轴距车辆原点3m
            y=np.sin(vehicle_yaw) * 3.0
        )

        # 找到最近路径点
        nearest_wp, nearest_idx = self._find_nearest_waypoint(front_axle, waypoints)
        if nearest_wp is None:
            return 0.0

        # 计算横向误差（车辆前轴到路径的垂直距离）
        path_vector = np.array([
            waypoints[nearest_idx + 1].transform.location.x - nearest_wp.transform.location.x,
            waypoints[nearest_idx + 1].transform.location.y - nearest_wp.transform.location.y
        ])
        path_angle = np.arctan2(path_vector[1], path_vector[0])
        vehicle_vector = np.array([
            front_axle.x - nearest_wp.transform.location.x,
            front_axle.y - nearest_wp.transform.location.y
        ])
        cross_track_error = np.linalg.norm(vehicle_vector) * np.sin(
            np.arctan2(vehicle_vector[1], vehicle_vector[0]) - path_angle
        )

        # Stanley控制公式
        heading_error = (path_angle - vehicle_yaw) % (2 * np.pi)
        if heading_error > np.pi:
            heading_error -= 2 * np.pi

        steer_angle = heading_error + np.arctan2(self.k_gain * cross_track_error, self.k_soft + current_speed)

        # 限制转向范围
        print(f"Steer: {steer_angle:.2f} \t| {heading_error:.2f} \t| {cross_track_error:.2f}")
        # breakpoint()
        return np.clip(-steer_angle, -self.max_steer, self.max_steer)

    def _find_nearest_waypoint(self, location, waypoints):
        min_dist = float('inf')
        nearest_wp = None
        nearest_idx = 0
        for i, wp in enumerate(waypoints[:-1]):
            dist = np.sqrt(
                (location.x - wp.transform.location.x)**2 +
                (location.y - wp.transform.location.y)**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_wp = wp
                nearest_idx = i
        return nearest_wp, nearest_idx