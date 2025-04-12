import math
import carla


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def run_step(
    waypoints_queue,
    vehicle,
    controller,
    debug=True,
    base_min_distance=3.0,
    distance_ratio=0.5,
):
    """
    Execute one step of local planning which involves running the longitudinal and lateral controllers to follow the waypoints trajectory.

    :param debug: boolean flag to activate waypoints debugging
    :return: control to be applied
    """
    target_waypoint = None

    # Purge the queue of obsolete waypoints
    veh_location = vehicle.get_location()
    vehicle_speed = get_speed(vehicle) / 3.6
    min_distance = base_min_distance + distance_ratio * vehicle_speed

    num_waypoint_removed = 0
    for waypoint, _ in waypoints_queue:

        if len(waypoints_queue) - num_waypoint_removed == 1:
            min_distance = 1  # Don't remove the last waypoint until very close by
        else:
            min_distance = min_distance

        if veh_location.distance(waypoint.transform.location) < min_distance:
            num_waypoint_removed += 1
        else:
            break

    if num_waypoint_removed > 0:
        for _ in range(num_waypoint_removed):
            waypoints_queue.popleft()

    # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
    if len(waypoints_queue) == 0:
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
    else:
        target_waypoint, velocity = waypoints_queue[0]
        waypoints = [waypoint[0] for waypoint in list(waypoints_queue)]
        control = controller.run_step(velocity, waypoints)

    if debug and target_waypoint is not None:
        draw_waypoints(vehicle.get_world(), [target_waypoint], 1.0)

    return control


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
