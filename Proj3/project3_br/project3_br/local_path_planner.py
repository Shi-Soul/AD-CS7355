from math_utils import *
from map_utils import *
import occupancy_grid
import path_finding_algorithms
import path_optimizer
import collision_checker
import carla
from heapq import heappush, heappop
import time
import math  # 确保导入了math模块

def normalize_angle(angle):
    """
    将角度归一化到 [-pi, pi] 范围内
    
    Args:
        angle: 输入角度（弧度）
    
    Returns:
        float: 归一化后的角度（弧度）
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

class LocalPlanner:
    def __init__(self, global_path):
        self._global_path = global_path
        self._goal_point = global_path[-1]
        self._path_searcher = path_finding_algorithms.PathFindingAlgorithm()
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = collision_checker.CollisionChecker()
        self._resolution = 0.2 # 每个网格单元格的分辨率（米）
        self._grid_len = 120 # 占用地图的长度和高度（米）
        self._grid_size = round(self._grid_len / self._resolution)
        self._grid_center_x = self._grid_size // 2
        self._grid_center_y = self._grid_size // 2
        self._occupancy_grid = None
        self._local_path = None
        self._search_count = 0
        
    def create_occupancy_grid(self, world, carla_map, ego_position, ego_heading):
        self._occupancy_grid = occupancy_grid.OccupancyGrid(self._grid_size, self._resolution, (self._grid_center_x, self._grid_center_y))
        
        # 强制边界约束（车道边界）
        # ------------------------------------------------------------------
        boundary_offset = 6.0  # 基础边界偏移量
        
        waypoints = carla_map.generate_waypoints(distance=2.0)
        
        waypoints_ = []
        for waypoint in waypoints:
            waypoint_location = waypoint.transform.location
            if ((ego_position.x - waypoint_location.x) ** 2 + (ego_position.y - waypoint_location.y) ** 2) ** 0.5 < self._grid_len / 2.0 * math.sqrt(2):
                waypoints_.append(waypoint)
                
        # 检查是否需要超车
        need_overtake = False
        for actor in world.get_actors():
            if 'vehicle' in actor.type_id and actor.attributes.get('role_name') != 'ego_vehicle':
                car_transform = actor.get_transform()
                car_position = car_transform.location
                dx = car_position.x - ego_position.x
                dy = car_position.y - ego_position.y
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < 30.0 and distance > 5.0:
                    angle = math.atan2(dy, dx) - ego_heading
                    angle = normalize_angle(angle)
                    if abs(angle) < math.pi/4:
                        need_overtake = True
                        break
        
        if need_overtake:
            # 超车时增加边界偏移量
            boundary_offset = 8.0  # 增大边界偏移量
            print("检测到需要超车，增加边界偏移量到", boundary_offset, "米")
        
        # 根据中心车道路点计算道路边界
        for waypoint in waypoints_:
            # 获取车道类型
            lane_type = waypoint.lane_type
            if lane_type == carla.LaneType.Driving:
                if waypoint.lane_id < 0:  # 逆向车道
                    # 超车时给逆向车道留出更多空间
                    if need_overtake:
                        vertices = create_rotated_rectangle_offset_in_place(
                            waypoint.transform.location, 
                            waypoint.lane_width / 2.0, 
                            waypoint.lane_width / 2.0,
                            boundary_offset * 1.5,  # 进一步增大逆向车道的边界偏移
                            boundary_offset / 2.0,
                            waypoint.transform.rotation.yaw * np.pi / 180
                        )
                    else:
                        vertices = create_rotated_rectangle_offset_in_place(
                            waypoint.transform.location, 
                            waypoint.lane_width / 2.0, 
                            waypoint.lane_width / 2.0,
                            boundary_offset,
                            boundary_offset / 2.0,
                            waypoint.transform.rotation.yaw * np.pi / 180
                        )
                else:  # 正向车道
                    vertices = create_rotated_rectangle_offset_in_place(
                        waypoint.transform.location, 
                        waypoint.lane_width / 2.0, 
                        waypoint.lane_width / 2.0,
                        boundary_offset / 2.0,
                        boundary_offset / 2.0,
                        waypoint.transform.rotation.yaw * np.pi / 180
                    )
            vertices = global_2_local_coord_vertices(vertices, ego_position, ego_heading, self._resolution, self._grid_center_x, self._grid_center_y)
            self._occupancy_grid.mark_grid_polygon(vertices, 0)
        # ------------------------------------------------------------------
        
        # 使用停放的车辆更新网格
        # ------------------------------------------------------------------
        for actor in world.get_actors():
            if 'vehicle' in actor.type_id:
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    continue  # 跳过自车
                else:
                    car_transform = actor.get_transform()
                    car_position = car_transform.location
                    car_extent = actor.bounding_box.extent
                    car_heading = car_transform.rotation.yaw * np.pi / 180  # 转换为弧度
                    vertices = create_rotated_rectangle_in_place(car_position, car_extent.y * 2, car_extent.x * 2, car_heading)
                    vertices = global_2_local_coord_vertices(vertices, ego_position, ego_heading, self._resolution, self._grid_center_x, self._grid_center_y)
                    self._occupancy_grid.mark_grid_polygon(vertices, 1)
        # ------------------------------------------------------------------ 
        
    # 查找路径中最接近当前位置的索引的函数
    def find_closest_index(self, current_position, path):
        min_distance = float('inf')
        closest_index = 0
        for i, point in enumerate(path):
            distance = math.sqrt((current_position.x - point[0])**2 + (current_position.y - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index
    
    # TODO: 您应该自定义自己的偏移量（偏移量需要小于60米,注意过小的偏移量会导致局部路径过短而频繁重规划）
    def choose_goal_cell(self, ego_position, ego_heading, world, offset = 20.0):
        """
        选择占用地图中的目标单元格。
        
        Args:
            ego_position: 自车的当前位置
            ego_heading: 自车的当前朝向（弧度）
            world: CARLA世界对象
            offset: 自车与目标点之间的距离（米）
        
        Returns:
            tuple: 局部坐标系中的目标单元格 (x, y, heading)
        """
        
        # 1. 查找全局路径上最接近的点的索引
        current_index = self.find_closest_index(ego_position, self._global_path)
        path_points_separation = 1.0 # 米

        # 2. 利用偏移量找到目标索引
        goal_index = current_index + int(offset / path_points_separation)
        if goal_index >= len(self._global_path):
            goal_index = len(self._global_path) - 1
        
        # 3. 检查目标是否太靠近障碍物
        max_attempts = 5  # 最大尝试次数
        attempt = 0
        original_goal_index = goal_index # 保存原始计算出的goal_index
        
        # 检查前方是否有车辆需要超车
        need_overtake = False
        target_vehicle = None
        for actor in world.get_actors():
            if 'vehicle' in actor.type_id and actor.attributes.get('role_name') != 'ego_vehicle':
                car_transform = actor.get_transform()
                car_position = car_transform.location
                dx = car_position.x - ego_position.x
                dy = car_position.y - ego_position.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 30.0 and distance > 5.0:
                    angle = math.atan2(dy, dx) - ego_heading
                    angle = normalize_angle(angle)
                    if abs(angle) < math.pi/4:
                        need_overtake = True
                        target_vehicle = actor
                        break
        
        if need_overtake and target_vehicle:
            # 获取目标车辆的位置和朝向
            target_transform = target_vehicle.get_transform()
            target_position = target_transform.location
            target_heading = target_transform.rotation.yaw * np.pi / 180
            
            # 计算超车目标点
            # 在目标车辆前方20米处，向左偏移3米
            overtake_offset = 20.0  # 超车目标点距离
            lateral_offset = 3.0    # 横向偏移量
            
            # 计算目标点位置
            target_x = target_position.x + overtake_offset * math.cos(target_heading)
            target_y = target_position.y + overtake_offset * math.sin(target_heading)
            
            # 向左偏移
            target_x += lateral_offset * math.cos(target_heading + math.pi/2)
            target_y += lateral_offset * math.sin(target_heading + math.pi/2)
            
            # 将目标点转换为局部坐标系
            goal_cell_coords = global_2_local_coord_point(
                (target_x, target_y, target_heading),
                ego_position,
                ego_heading,
                self._resolution,
                self._grid_center_x,
                self._grid_center_y
            )
            
            print("使用超车目标点:", (target_x, target_y, target_heading))
            return goal_cell_coords
        
        while attempt < max_attempts:
            # 确保goal_index在有效范围内
            current_checked_goal_index = np.clip(goal_index, 0, len(self._global_path) - 1)

            goal_point = self._global_path[current_checked_goal_index]
            # 将目标点转换为局部坐标系
            local_goal_for_check = global_2_local_coord_point(goal_point, ego_position, ego_heading, 
                                                  self._resolution, self._grid_center_x, self._grid_center_y)
            
            if self._collision_checker.collision_check(local_goal_for_check, self._occupancy_grid):
                goal_index = current_checked_goal_index
                break
                
            # 调整 goal_index 的逻辑
            if attempt == 0:
                pass
            elif attempt % 2 == 1:  # 奇数次尝试，向后
                goal_index = original_goal_index + (attempt // 2 + 1)
            else:  # 偶数次尝试，向前
                goal_index = original_goal_index - (attempt // 2)
            
            # 确保调整后的 goal_index 在有效范围内
            goal_index = np.clip(goal_index, 0, len(self._global_path) - 1)
            
            attempt += 1
        else:
            print(f"警告: 在 choose_goal_cell 中未能找到完全无碰撞的目标点，使用最后尝试的点 (index: {goal_index})")

        # 4. 将最终选择的（或最后尝试的）目标点转换为局部坐标系中的单元格
        final_goal_point_global = self._global_path[goal_index]
        goal_cell_coords = global_2_local_coord_point(final_goal_point_global, ego_position, ego_heading, 
                                                   self._resolution, self._grid_center_x, self._grid_center_y)

        return goal_cell_coords
    
    def find_local_path(self, world, carla_map, ego_position, ego_heading):
        # 寻找最优局部路径的路径搜索算法
        
        # 1. 构建占用地图
        self.create_occupancy_grid(world, carla_map, ego_position, ego_heading)
                        
        # 2. 在占用地图中选择目标单元格
        self._search_count += 1
        goal_cell = self.choose_goal_cell(ego_position, ego_heading, world)
        start_cell = path_finding_algorithms.Node(self._grid_center_x, self._grid_center_y, np.pi / 2.0)
        goal_cell = path_finding_algorithms.Node(goal_cell[0], goal_cell[1], goal_cell[2])
        
        # 在此处调用 save_test_case
        if self._occupancy_grid: # 确保占用栅格图已创建
            # save_test_case 需要起点和终点作为参数。
            # 这些点应该是 Hybrid A* 使用的栅格坐标。
            # start_cell 和 goal_cell 已经是 Node 对象，包含 x, y, heading
            # 我们传递它们的栅格坐标和航向。
            start_point_for_save = (start_cell.x, start_cell.y, start_cell.heading)
            goal_point_for_save = (goal_cell.x, goal_cell.y, goal_cell.heading)
            self._occupancy_grid.save_test_case(start_point_for_save, goal_point_for_save)
            print(f"DEBUG: Test case saved with start: {start_point_for_save}, goal: {goal_point_for_save}")


        # 3. 搜索局部路径
        local_path_in_occupancy_map = self._path_searcher.search_path(start_cell, goal_cell, self._occupancy_grid, self._search_count)
        
        # 4. 从路径中采样点并使用样条拟合优化轨迹
        sampled_local_path_in_occupancy_map = self._path_optimizer.sample_path(local_path_in_occupancy_map)
        smooth_local_path_in_occupancy_map = self._path_optimizer.smooth_path_spline(sampled_local_path_in_occupancy_map)
        
        # 5. 将路径点转换为全局坐标系
        local_path = []
        for point in smooth_local_path_in_occupancy_map:
            local_point_in_global = local_2_global_coord_point(point, ego_position, ego_heading, 
                                                               self._resolution, self._grid_center_x, self._grid_center_y)
            local_path.append((local_point_in_global[0], local_point_in_global[1], (point[2] + ego_heading - np.pi / 2.0) / np.pi * 180.0))
            
        self._local_path = local_path

    # TODO: 您应该自定义自己的距离阈值 (hint: 距离阈值过短，会导致频繁重规划)
    def need_to_recreate_occupancy_grid(self, ego_position, distance_threshold = 5.0):
        """
        判断是否需要重新创建占用栅格图和规划新的局部路径。
        
        此函数在每次控制循环中被调用，用于判断当前局部路径是否仍然适用。
        当以下条件之一满足时，将返回True表示需要重新规划：
        1. 局部路径或占用栅格图尚未创建
        2. 自车过于接近局部路径末端
        
        Args:
            ego_position: 自车当前位置
            distance_threshold: 判断是否需要重新规划的距离阈值，默认为5.0米，调整为10.0米
        
        Returns:
            bool: 如果需要重新创建占用栅格图和规划路径则返回True，否则返回False
        """
        # 修改重规划判断逻辑
        if self._local_path == None or self._occupancy_grid == None or len(self._local_path) == 0:
            return True
        
        # 特殊情况：如果自车已接近全局路径终点，不需要重新规划
        if math.sqrt((ego_position.x - self._goal_point[0])**2 + \
                     (ego_position.y - self._goal_point[1])**2) <= distance_threshold + 3.0:
            return False

        # 增加判断条件：如果当前路径仍然可行，不要重新规划
        # 检查自车到局部路径的距离
        min_distance_to_path = float('inf')
        for point in self._local_path:
            distance = math.sqrt((ego_position.x - point[0])**2 + \
                               (ego_position.y - point[1])**2)
            min_distance_to_path = min(min_distance_to_path, distance)
        
        # 如果自车距离路径太远（超过2米），才需要重新规划
        if min_distance_to_path > 2.0:
            return True
        
        # 如果自车太靠近当前局部路径的末端点，才需要重新规划
        if math.sqrt((ego_position.x - self._local_path[-1][0])**2 + \
                     (ego_position.y - self._local_path[-1][1])**2) <= distance_threshold:
            return True
        
        return False
    
    # TODO: 您应该自定义自己的前瞻距离
    def find_lookahead_point(self, world, carla_map, ego_position, ego_heading, lookahead_len = 6.0):
        """
        在当前局部路径上找到适合的前瞻点，用于车辆控制。
        
        前瞻点是局部路径上距离自车lookahead_len距离的点，用于车辆跟踪控制。
        如果当前局部路径需要重新规划，会先调用find_local_path重新规划。
        
        Args:
            world: CARLA世界对象，用于访问仿真环境
            carla_map: CARLA地图对象，用于获取道路信息
            ego_position: 自车当前位置
            ego_heading: 自车当前航向
            lookahead_len: 前瞻距离，默认为1.0米
        
        Returns:
            tuple: 前瞻点信息 (x, y, yaw)，包含位置和航向
        """
        print(f"\n[DEBUG] 开始寻找前瞻点:")
        print(f"当前自车位置: ({ego_position.x:.2f}, {ego_position.y:.2f}), 航向: {ego_heading:.2f} rad")
        print(f"前瞻距离: {lookahead_len:.2f} 米")
        
        # 检查是否需要重新规划局部路径
        if self.need_to_recreate_occupancy_grid(ego_position):
            print("需要重新规划局部路径...")
            self.find_local_path(world, carla_map, ego_position, ego_heading)
        else:
            print("使用现有局部路径")
        
        if self._local_path is None or len(self._local_path) == 0:
            print("警告: 局部路径为空!")
            return None
            
        print(f"局部路径长度: {len(self._local_path)} 个点")
        
        # 找到局部路径上离自车最近的点的索引
        closest_index = self.find_closest_index(ego_position, self._local_path)
        print(f"最近点索引: {closest_index}")
        print(f"最近点坐标: ({self._local_path[closest_index][0]:.2f}, {self._local_path[closest_index][1]:.2f})")
        
        # 从最近点开始，沿着路径累计距离，找到距离为lookahead_len的点
        distance = 0
        for i in range(closest_index + 1, len(self._local_path)):
            # 计算当前路径段长度并累加
            distance += math.sqrt((self._local_path[i][0] - self._local_path[i - 1][0])**2 + \
                                  (self._local_path[i][1] - self._local_path[i - 1][1])**2)
            # 当累计距离超过lookahead_len时，找到合适的前瞻点
            if distance > lookahead_len:
                print(f"找到前瞻点:")
                print(f"索引: {i}")
                print(f"坐标: ({self._local_path[i][0]:.2f}, {self._local_path[i][1]:.2f})")
                print(f"航向: {self._local_path[i][2]:.2f} 度")
                print(f"累计距离: {distance:.2f} 米")
                return self._local_path[i]
        
        # 如果路径上没有点距离超过lookahead_len，返回路径末端点
        print("未找到满足距离要求的前瞻点，返回路径末端点")
        print(f"末端点坐标: ({self._local_path[-1][0]:.2f}, {self._local_path[-1][1]:.2f})")
        print(f"末端点航向: {self._local_path[-1][2]:.2f} 度")
        return self._local_path[-1]