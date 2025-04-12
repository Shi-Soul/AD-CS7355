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
    """
    Custom longitudinal controller for controlling the vehicle's speed.

    Attributes:
        vars (CUtils): Utility object for managing persistent variables.
    """

    def __init__(self, vehicle):
        """
        Initializes the CustomLongitudinalController.

        :param vehicle: The vehicle object to control.
        """
        self.vars = CUtils()
        # DECLARE USAGE VARIABLES HERE (Use self.vars.create_var(<variable name>, <default value>) to create a persistent variable.)

    def run_step(self, target_speed):
        """
        Executes one step of the longitudinal control logic to adjust the vehicle's speed.

        :param target_speed: The desired speed for the vehicle to maintain in km/h.
        :return: Throttle/brake control in the range [-1, 1], where:
            -1 represents maximum braking,
            1 represents maximum acceleration.

        """
        # Your Custom Longitudinal Control Logic Here
        pass


class CustomLateralController:
    """
    Custom lateral controller for controlling the vehicle's steering.

    Attributes:
        vars (CUtils): Utility object for managing persistent variables.
    """

    def __init__(self, vehicle):
        """
        Initializes the CustomLateralController.

        :param vehicle: The vehicle object to control.
        """
        self.vars = CUtils()
        # DECLARE USAGE VARIABLES HERE (Use self.vars.create_var(<variable name>, <default value>) to create a persistent variable.)

    def run_step(self, waypoints):
        """
        Executes one step of the lateral control logic to adjust the vehicle's steering.

        :param waypoints: A list of waypoints for the vehicle to follow, each waypoint is a carla.Waypoint object.
        :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            1 maximum steering to right
        """
        # Your Custom Lateral Control Logic Here
        pass
