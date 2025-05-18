from math_utils import *
from map_utils import *
import occupancy_grid
import path_finding_algorithms
import path_optimizer
import collision_checker
import carla
from heapq import heappush, heappop
import time

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
        boundary_offset = 5.0  # 定义边界偏移量，根据需要调整
        
        waypoints = carla_map.generate_waypoints(distance=2.0)
        
        waypoints_ = []
        for waypoint in waypoints:
            waypoint_location = waypoint.transform.location
            if ((ego_position.x - waypoint_location.x) ** 2 + (ego_position.y - waypoint_location.y) ** 2) ** 0.5 < self._grid_len / 2.0 * math.sqrt(2):
                waypoints_.append(waypoint)
                
        # 根据中心车道路点计算道路边界
        for waypoint in waypoints_:
            vertices = create_rotated_rectangle_offset_in_place(waypoint.transform.location, waypoint.lane_width / 2.0, waypoint.lane_width / 2.0,
                                                                boundary_offset / 2.0, boundary_offset / 2.0, 
                                                                waypoint.transform.rotation.yaw * np.pi / 180)
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
    def choose_goal_cell(self, ego_position, ego_heading, offset = 1.0):
        """
        选择占用地图中的目标单元格。
        
        此函数首先在全局路径上找到距离自车位置为offset的点，
        然后检查该点是否远离障碍物，最后将其转换为局部坐标系中的单元格。
        
        Args:
            ego_position: 自车的当前位置
            ego_heading: 自车的当前朝向（弧度）
            offset: 自车与目标点之间的距离（米）
        
        Returns:
            tuple: 局部坐标系中的目标单元格 (x, y, heading)
        """
        
        # 1. 查找全局路径上最接近的点的索引
        current_index = self.find_closest_index(ego_position, self._global_path)
        path_points_separation = 1.0 # 米

        # 2. TODO: 利用偏移量找到目标索引
        # (提示：全局路径上两点之间的距离是path_points_separation，
        # 自车位置与目标点之间的距离是offset，
        # 因此目标索引可以直接计算)
        goal_index = current_index + ...
        if goal_index >= len(self._global_path):
            goal_index = len(self._global_path) - 1
        
        # 3. TODO: 检查目标是否太靠近障碍物。
        # 如果太近，通过调整goal_index尝试找到一个更近或更远的目标点。
        # 您可以使用collision_checker.py文件中的collision_check()函数检查目标是否太靠近障碍物。
        # (提示：需要先将目标点转换为局部坐标系，然后调用self._collision_checker进行碰撞检查)
        # 参考实现：
        # 1. 将目标点转换为占用栅格中的坐标 (global_2_local_coord_point函数)
        # 2. 使用self._collision_checker.collision_check()检查目标点是否太靠近障碍物
        # 3. 如果太靠近，调整goal_index，可前后尝试（例如goal_index+1, goal_index-1, goal_index+2...）


        # 4. 将目标点转换为局部坐标系中的单元格
        goal_cell = global_2_local_coord_point(self._global_path[goal_index], ego_position, ego_heading, 
                                                   self._resolution, self._grid_center_x, self._grid_center_y)

        return goal_cell
    
    def find_local_path(self, world, carla_map, ego_position, ego_heading):
        # 寻找最优局部路径的路径搜索算法
        
        # 1. 构建占用地图
        self.create_occupancy_grid(world, carla_map, ego_position, ego_heading)
                        
        # 2. 在占用地图中选择目标单元格
        self._search_count += 1
        goal_cell = self.choose_goal_cell(ego_position, ego_heading)
        start_cell = path_finding_algorithms.Node(self._grid_center_x, self._grid_center_y, np.pi / 2.0)
        goal_cell = path_finding_algorithms.Node(goal_cell[0], goal_cell[1], goal_cell[2])
        
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
            distance_threshold: 判断是否需要重新规划的距离阈值，默认为5.0米
        
        Returns:
            bool: 如果需要重新创建占用栅格图和规划路径则返回True，否则返回False
        """
        # 情况1：如果局部路径或占用栅格图尚未创建，则需要重新规划
        if self._local_path == None or self._occupancy_grid == None or len(self._local_path) == 0:
            return True
        
        # 特殊情况：如果自车已接近全局路径终点，不需要重新规划
        # distance_threshold + 3.0作为缓冲区，避免频繁重规划
        if math.sqrt((ego_position.x - self._goal_point[0])**2 + \
                     (ego_position.y - self._goal_point[1])**2) <= distance_threshold + 3.0:
            return False

        # 情况2：如果自车太靠近当前局部路径的末端点，则需要重新规划
        # 计算自车到局部路径末端点的欧氏距离，如果小于阈值则需要重新规划
        if math.sqrt((ego_position.x - self._local_path[-1][0])**2 + \
                     (ego_position.y - self._local_path[-1][1])**2) <= distance_threshold:
            return True
        
        # 默认情况：当前局部路径仍然可用，不需要重新规划
        return False
    
    # TODO: 您应该自定义自己的前瞻距离
    def find_lookahead_point(self, world, carla_map, ego_position, ego_heading, lookahead_len = 1.0):
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
        # 检查是否需要重新规划局部路径
        # 如果需要，先调用find_local_path生成新的局部路径
        if self.need_to_recreate_occupancy_grid(ego_position):
            self.find_local_path(world, carla_map, ego_position, ego_heading)
        
        # 找到局部路径上离自车最近的点的索引
        closest_index = self.find_closest_index(ego_position, self._local_path)
        
        # 从最近点开始，沿着路径累计距离，找到距离为lookahead_len的点
        distance = 0
        for i in range(closest_index + 1, len(self._local_path)):
            # 计算当前路径段长度并累加
            distance += math.sqrt((self._local_path[i][0] - self._local_path[i - 1][0])**2 + \
                                  (self._local_path[i][1] - self._local_path[i - 1][1])**2)
            # 当累计距离超过lookahead_len时，找到合适的前瞻点
            if distance > lookahead_len:
                return self._local_path[i]
        
        # 如果路径上没有点距离超过lookahead_len，返回路径末端点
        return self._local_path[-1]