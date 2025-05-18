import carla
import math

def spawn_ego_vehicle(world, ego_state):
    try:
        spawn_point = carla.Transform(
            carla.Location(x = ego_state[0], y = ego_state[1], z = 1.999070), 
            carla.Rotation(roll = 0.0, pitch = 0.0, yaw = ego_state[2]))
        blueprint_library = world.get_blueprint_library()
        ego_vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        ego_vehicle_bp.set_attribute('role_name', 'ego_vehicle')
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)
        return ego_vehicle
    except RuntimeError:
        print("can't spwan ego vehicle")
        
def spawn_parked_vehicles(world, exo_states):
    parked_vehicle_trans = [carla.Transform(
                            carla.Location(x = exo_state[0], y = exo_state[1], z = 1.999070), 
                            carla.Rotation(roll = 0.0, pitch = 0.0, yaw = exo_state[2]))
                            for exo_state in exo_states]
    vehicles = []
    try:
        for tran_point in parked_vehicle_trans:
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.audi.a2')
            vehicle_bp.set_attribute('role_name', 'parked_vehicle')
            vehicle = world.spawn_actor(vehicle_bp, tran_point)
            vehicles.append(vehicle)
        return vehicles
    except RuntimeError:
        print("can't spwan parked vehicle")