import os
import argparse
import logging
import carla
import numpy as np
import local_path_planner
import controller
import math_utils
import map_utils
import weakref
import math
from carla import ColorConverter as cc
import datetime

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def write_trajectory_file(task_index, x_list, y_list, theta_list, v_list, acc_list, yaw_rate_list, t_list):
    create_controller_output_dir(os.path.dirname(os.path.realpath(__file__)) + f'/task2_output/output_{task_index}/')
    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)) + f'/task2_output/output_{task_index}/', 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write(f'{x_list[i]}, {y_list[i]}, {theta_list[i]}, {v_list[i]}, {acc_list[i]}, {yaw_rate_list[i]}, {t_list[i]}\n')

def read_waypoints(file_path):
    waypoints_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             file_path)
    waypoints = []
    with open(waypoints_filepath) as waypoints_file_handle:
        lines = waypoints_file_handle.readlines()
        for line in lines:
            waypoint = line.split(', ')
            if len(waypoint) > 0:
                waypoints.append((float(waypoint[0]), float(waypoint[1]), np.deg2rad(float(waypoint[2]))))
    return waypoints

def read_vehicles(file_path):
    vehicle_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             file_path)
    vehicle_states = []
    with open(vehicle_filepath) as vehicle_file_handle:
        lines = vehicle_file_handle.readlines()
        for line in lines:
            state = line.split(', ')
            if len(state) > 0:
                vehicle_states.append((float(state[0]), float(state[1]), float(state[2])))
    return vehicle_states

def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

def get_actor_display_name(actor, truncate=250):
    """获取角色显示名称的方法"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def spawn_collision_sensor(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'),
                                    carla.Transform(), attach_to=vehicle)
    def record_collsion(event):
        # 如果发生碰撞，停止程序
        collision_sensor.stop()
        print(f'与车辆{event.other_actor.id}发生碰撞，规划器停止')
        exit()
    collision_sensor.listen(lambda event: record_collsion(event))
    return collision_sensor

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """ 渐变文本类 """

    def __init__(self, font, dim, pos):
        """构造方法"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """设置渐变文本"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """每个时钟周期的渐变文本方法"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """渲染渐变文本方法"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    """ 文本渲染帮助类 """

    def __init__(self, font, width, height):
        """构造方法"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """切换显示或隐藏帮助"""
        self._render = not self._render

    def render(self, display):
        """渲染帮助文本方法"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """HUD文本类"""

    def __init__(self, width, height):
        """构造方法"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """每个时钟周期从世界获取信息"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD方法每个时钟周期"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """切换信息显示或隐藏"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """通知文本"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """错误文本"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """渲染HUD类"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ 相机管理类 """

    def __init__(self, parent_actor, hud):
        """构造方法"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """激活相机"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """设置传感器"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # 我们需要向lambda传递对self的弱引用
            # 以避免循环引用
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """获取下一个传感器"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """切换录制开关"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """渲染方法"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# Main function to control the car
def control_car(args):
    pygame.init()
    pygame.font.init()
    ego_vehicle = None
    parked_vehicles = None
    sensors = []
    client = None
    sim_world = None
    camera_manager = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        traffic_manager = client.get_trafficmanager()
        client.load_world('Town02')
        sim_world = client.get_world()
        carla_map = sim_world.get_map()

        # 设置同步模式
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        #settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)

        # 读取车辆状态
        directory_name = f'task2/test_case{args.task_index}'
        vehicle_states = read_vehicles(directory_name + "/vehicle.txt")
        
        # 生成自车和其他车辆
        ego_vehicle = map_utils.spawn_ego_vehicle(sim_world, vehicle_states[0])
        parked_vehicles = map_utils.spawn_parked_vehicles(sim_world, vehicle_states[1:])
        sim_world.tick()

        gameDisplay = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        
        camera_manager = CameraManager(ego_vehicle, hud)
        camera_manager.transform_index = 0
        camera_manager.set_sensor(0, notify=False)
        
        # 设置观察者
        spectator = sim_world.get_spectator()
        
        # 读取全局路径
        gloabl_path = read_waypoints(directory_name + "/global_path.txt")
        
        # 创建局部规划器
        local_path_planner_ = local_path_planner.LocalPlanner(gloabl_path)
        
        # 创建控制器
        controller_ = controller.Controller(ego_vehicle)
        
        x_history        = []
        y_history        = []
        yaw_history      = []
        speed_history    = []
        acc_history      = []
        yaw_rate_history = []
        time_history     = []
        sensors.append(spawn_collision_sensor(sim_world, ego_vehicle))
        while True:
            camera_manager.render(gameDisplay)
            pygame.display.flip()
            
            ego_transform = ego_vehicle.get_transform()
            ego_position = ego_transform.location
            ego_heading = ego_transform.rotation.yaw * np.pi / 180  # 转换为弧度
            ego_speed = ego_vehicle.get_velocity().length()
            ego_acc = ego_vehicle.get_acceleration().length()
            ego_yaw_rate = ego_vehicle.get_angular_velocity().z * np.pi / 180  # 转换为弧度
            
            # 存储历史数据
            x_history.append(ego_position.x)
            y_history.append(ego_position.y)
            yaw_history.append(ego_heading)
            speed_history.append(ego_speed)
            acc_history.append(ego_acc)
            yaw_rate_history.append(ego_yaw_rate)
            time_history.append(sim_world.get_snapshot().timestamp.elapsed_seconds)
            
            # 如果车辆到达目标
            if math_utils.dist2d((ego_position.x, ego_position.y), gloabl_path[-1]) <= 2.0:
                break

            # 默认直线行驶
            control = carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0, hand_brake=False, reverse=False)
            ego_vehicle.apply_control(control)
            sim_world.tick()
            continue
            
            # 获取前视点或控制用的未来路径
            look_ahead_point = local_path_planner_.find_lookahead_point(sim_world, carla_map, ego_position, ego_heading)
            target_location = carla.Location(x = look_ahead_point[0], y = look_ahead_point[1], z = 0.0)
            target_rotation = carla.Rotation(roll = 0.0, pitch = 0.0, yaw = look_ahead_point[2])
            control = controller_.run_step(target_location, target_rotation)
            ego_vehicle.apply_control(control)
            #sim_world.wait_for_tick()
            sim_world.tick()
            
        write_trajectory_file(args.task_index, x_history, y_history, yaw_history, speed_history, acc_history, yaw_rate_history, time_history)
    
    # except Exception as e:
    #     pass
        
    finally:
        if client is not None:
            
            if ego_vehicle is not None:
                # 销毁自车
                ego_vehicle.destroy()
            
            if parked_vehicles is not None:
                # 销毁停放的车辆
                client.apply_batch([carla.command.DestroyActor(x) for x in parked_vehicles])
            
            # 销毁传感器
            for sensor in sensors:
                if sensor is not None:
                    sensor.destroy()
                
            camera_manager.sensor.destroy()
        
        if sim_world is not None:
            settings = sim_world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            sim_world.apply_settings(settings)
        pygame.quit()

def main():
    """主方法"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--task-index',
        metavar='N',
        type=int,
        default=0,
        help='Index of the task (default: 0)'
    )
    argparser.add_argument(
        '-t',
        metavar='N',
        type=float,
        default=0.05,
        help='Fixed time-step'
    )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        control_car(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')  

if __name__ == '__main__':
    main()
