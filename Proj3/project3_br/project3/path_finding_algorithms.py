import dubins
from heapq import heappush, heappop
import occupancy_grid
import collision_checker
import math_utils
import math
import numpy as np
import time
from typing import Tuple, List, Dict, Any

# 本地定义的角度归一化函数
def normalize_angle(angle):
    """将角度归一化到 [-pi, pi) 区间"""
    while angle >= math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

# 航向网格分辨率
YAW_GRID_RESOLUTION = np.deg2rad(5.0)

class Node:
    def __init__(self, x, y, heading, g_cost = 0, h_cost = 0, predecssor = None) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.g_cost = g_cost # 已经走过的路径代价（实际值）
        self.h_cost = h_cost # 到目标的估计代价（启发式值）
        self.predecssor = predecssor
        
    def CalIndex1DWithAngle(self, width, height):
        min_heading_index = math.floor(-math.pi / YAW_GRID_RESOLUTION)
        heading_index = math.floor(self.heading / YAW_GRID_RESOLUTION)
        # Ensure indices are within valid discrete grid bounds after floor operation
        x_idx = max(0, min(width - 1, math.floor(self.x)))
        y_idx = max(0, min(height - 1, math.floor(self.y)))
        return (heading_index - min_heading_index) * width * height \
                + y_idx * width + x_idx

    def CalIndex1D(self, width, height):
         # Ensure indices are within valid discrete grid bounds after floor operation
        x_idx = max(0, min(width - 1, math.floor(self.x)))
        y_idx = max(0, min(height - 1, math.floor(self.y)))
        return y_idx * width + x_idx

    def CalIndex2D(self):
         # Ensure indices are integer grid coordinates
        return (math.floor(self.x), math.floor(self.y))

def GenerateSuccessors(node: Node, node_id: int, goal_node: Node) -> List[Node]:
    """
    生成后继节点列表，应用车辆运动学模型产生有效的后继状态。
    
    Args:
        node: 当前节点，包含位置和航向信息
        node_id: 当前节点的唯一标识符
        goal_node: 目标节点，用于计算距离
    
    Returns:
        list: 可能的后继节点列表
    """
    successors = []
    
    # 计算到目标点的距离
    distance_to_goal = math.sqrt((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)
    
    # 根据距离动态调整转向角度
    if distance_to_goal > 10.0:  # 距离较远时使用较小转向角度，保持稳定
        steer_angles_deg = [-15.0, -7.5, 0.0, 7.5, 15.0]
    else:  # 距离较近时使用较大转向角度，提高机动性
        steer_angles_deg = [-30.0, -15.0, 0.0, 15.0, 30.0]
    
    # 根据距离动态调整移动距离
    # 距离远时步长可以大一些，距离近时步长要小一些以提高精度
    distance = max(0.5, min(2.0, distance_to_goal * 0.1))
    
    # 将角度转换为弧度
    delta_thetas = [np.deg2rad(angle) for angle in steer_angles_deg]

    for delta_theta in delta_thetas:
        # 计算新的位置和航向
        new_heading = normalize_angle(node.heading + delta_theta)
        new_x = node.x + distance * math.cos(new_heading)
        new_y = node.y + distance * math.sin(new_heading)

        # 计算新节点的g_cost
        move_cost = distance
        if abs(delta_theta) > 1e-6:  # 检查是否转弯
            # 转弯惩罚系数也根据距离动态调整
            turn_penalty = 1.5 if distance_to_goal > 10.0 else 1.2
            move_cost *= turn_penalty
            
        new_g_cost = node.g_cost + move_cost

        # 创建新的后继节点
        successors.append(Node(new_x, new_y, new_heading, new_g_cost, 0, node_id))
        
    return successors

def CalAStarPathCost(start_node: Node, h_map: dict, occupancy_map: occupancy_grid.OccupancyGrid) -> float:
    """
    计算从起始节点到目标节点的A*路径代价。
    
    Args:
        start_node: 起始节点
        h_map: 由holonomic_heuristic_Astar生成的启发式地图
    
    Returns:
        float: A*路径代价，如果找不到路径则返回无穷大
    """
    map_width, map_height = occupancy_map.get_grid_size(), occupancy_map.get_grid_size()
    node_idx_1d = start_node.CalIndex1D(map_width, map_height)
    if node_idx_1d not in h_map:
        return float('inf') # 使用无穷大表示不可达
    return h_map[node_idx_1d].g_cost # 返回存储在节点中的 g_cost

def CalDubinPathCost(start_node: Node, goal_node: Node) -> float:
    """
    计算从起始节点到目标节点的Dubins路径代价。
    
    Args:
        start_node: 起始节点
        goal_node: 目标节点
    
    Returns:
        float: Dubins路径代价，如果找不到有效路径则返回无穷大
    """
    #TODO: path = dubins.shortest_path(q0, q1, turning_radius), 你可以通过调节第三个参数控制转弯半径
    turning_radius = 3.0 # 调整转弯半径
    try:
        # Ensure headings are within [-pi, pi] for Dubins library if necessary
        q0 = (start_node.x, start_node.y, normalize_angle(start_node.heading))
        q1 = (goal_node.x, goal_node.y, normalize_angle(goal_node.heading))
        dubins_path = dubins.shortest_path(q0, q1, turning_radius)
        dubins_cost = dubins_path.path_length()
        # Add a small penalty for Dubins paths to slightly prefer grid search? (Optional)
        # dubins_cost *= 1.001
        return dubins_cost
    except ValueError: # 如果起点和终点太近或角度不合适，dubins可能抛出异常
        return float('inf')
    
def CalHCost(start_node: Node, goal_node: Node, h_map: dict, occupancy_map: occupancy_grid.OccupancyGrid) -> float:
    """
    计算启发式代价，取A*代价和Dubins代价的最大值。
    
    Args:
        start_node: 起始节点
        goal_node: 目标节点
        h_map: 启发式地图
    
    Returns:
        float: 启发式代价
    """
    # 添加距离权重
    distance_weight = 1.2
    heading_weight = 0.8
    
    astar_cost = CalAStarPathCost(start_node, h_map, occupancy_map)
    dubin_cost = CalDubinPathCost(start_node, goal_node)
    
    # 计算航向差异的代价
    heading_diff = abs(normalize_angle(start_node.heading - goal_node.heading))
    heading_cost = heading_diff * heading_weight
    
    if astar_cost == float('inf') and dubin_cost == float('inf'):
        return float('inf')
    elif astar_cost == float('inf'):
        return dubin_cost * distance_weight + heading_cost
    elif dubin_cost == float('inf'):
        return astar_cost * distance_weight + heading_cost
    else:
        return max(astar_cost, dubin_cost) * distance_weight + heading_cost

def DubinShot(start_node: Node, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid) -> Tuple[bool, List[Any]]:
    turning_radius = 3.0 # 保持与CalDubinPathCost一致
    try:
        q0 = (start_node.x, start_node.y, normalize_angle(start_node.heading))
        q1 = (goal_node.x, goal_node.y, normalize_angle(goal_node.heading))
        dubins_path = dubins.shortest_path(q0, q1, turning_radius)

        step_size = 1.0 # 采样点间隔
        points, _ = dubins_path.sample_many(step_size)
        if not points: # Handle empty path case
            return False, None

        collision_checker_ = collision_checker.CollisionChecker()
        # Check collision for each sampled point
        for point in points:
             # Create a temporary Node for collision checking - NO, pass tuple directly
             # node_for_check = Node(point[0], point[1], point[2])
             # collision_check returns True if NO collision
             if not collision_checker_.collision_check(point, occupancy_map):
                return False, None # Collision detected

        # Check the final goal node as well, as sampling might miss it
        goal_tuple = (goal_node.x, goal_node.y, goal_node.heading)
        if not collision_checker_.collision_check(goal_tuple, occupancy_map):
             return False, None

        return True, points # No collision, return success and path points
    except ValueError:
        return False, None # Dubins path could not be generated

class PathFindingAlgorithm:
    def __init__(self, algorithm = "Hybrid-A*") -> None:
        self.algorithm = algorithm
        
    def search_path(self, start_node: Node, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid, search_count: int) -> List[Tuple[float, float, float]]:
        """
        根据指定的算法搜索从起始节点到目标节点的路径。
        
        Args:
            start_node: 起始节点
            goal_node: 目标节点
            occupancy_map: 占用栅格地图
            search_count: 搜索次数计数
        
        Returns:
            list: 路径点列表
        """
        if self.algorithm == "Hybrid-A*":
            return self.HybridAStar(start_node, goal_node, occupancy_map, search_count)
        
    def holonomic_heuristic_Astar(self, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid) -> Dict[int, Node]:
        """
        使用全向A*算法计算启发式地图。
        
        此函数计算从地图上每个点到目标的最短路径代价，忽略车辆航向约束，
        用于为Hybrid-A*提供启发式值。
        
        Args:
            goal_node: 目标节点
            occupancy_map: 占用栅格地图
        
        Returns:
            dict: 存储每个节点A*代价的字典
        """
        # motion: (dx, dy, cost)
        s2 = math.sqrt(2)
        motion = [(1, 0, 1.0),
                    (0, 1, 1.0),
                    (-1, 0, 1.0),
                    (0, -1, 1.0),
                    (1, 1, s2),
                    (-1, 1, s2),
                    (-1, -1, s2),
                    (1, -1, s2)]
        map_width, map_height = occupancy_map.get_grid_size(), occupancy_map.get_grid_size()
        
        # openset: 存储待探索的节点的字典，键为节点的一维索引（CalIndex1D），值为节点对象
        # 包含仍需评估的节点，每次迭代从中选择代价最小的节点进行扩展
        openset = {} 
        
        # closedset: 存储已探索过的节点的字典，键为节点的一维索引（CalIndex1D），值为节点对象
        # 包含已探索过的节点，防止重复探索，同时用于计算启发式值
        closedset = {}
        
        goal_node.g_cost = 0.0
        goal_node_idx_1d = goal_node.CalIndex1D(map_width, map_height)
        openset[goal_node_idx_1d] = goal_node
        
        # pq: 优先队列，按g值排序的节点列表，用于高效取出g值最小的节点
        # 存储元组 (g值, 节点的一维索引)，g值越小优先级越高
        pq = []
        heappush(pq, (0.0, goal_node_idx_1d)) # 推入 (cost, node_id)
        
        while pq:
            # TODO: 在占用图中扩展所有节点并将其g_cost存储在closedset中
            # 1. 从优先队列pq中取出g值最小的节点
            cost, current_node_id = heappop(pq)

            # 如果节点已经在closedset中，跳过
            if current_node_id in closedset:
                continue

            # 2. 如果队列为空，应该在循环开始时检查，如果pq为空则break
            # 在这里我们假设如果pq空了，则说明没有可达路径（理论上不应发生，除非目标不可达）

            # 3. 将节点从openset移到closedset
            current_node = openset.pop(current_node_id)
            closedset[current_node_id] = current_node

            # 4. 遍历所有可能的移动方向
            for dx, dy, move_cost in motion:
                next_x = current_node.x + dx
                next_y = current_node.y + dy
                next_node = Node(next_x, next_y, 0) # heading 在这里不重要

                # 检查边界
                if not occupancy_map.is_inside(next_node.CalIndex2D()):
                    continue

                # 检查障碍物
                # 注意：occupancy_map.is_occupied 返回 True 表示有障碍
                if occupancy_map.is_occupied(next_node.CalIndex2D()):
                    continue

                next_node_id = next_node.CalIndex1D(map_width, map_height)

                # 跳过已拓展过的节点
                if next_node_id in closedset:
                    continue

                # 5. 计算新节点的代价并更新openset和pq
                new_g_cost = current_node.g_cost + move_cost
                if next_node_id not in openset or new_g_cost < openset[next_node_id].g_cost:
                    next_node.g_cost = new_g_cost
                    next_node.predecssor = current_node_id # 记录前驱用于路径重建（虽然这里只关心cost）
                    openset[next_node_id] = next_node
                    heappush(pq, (new_g_cost, next_node_id))

        # 当队列为空时，搜索结束
        return closedset # 返回包含代价信息的 closedset
        
    def HybridAStar(self, start_node: Node, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid, search_count: int) -> List[Tuple[float, float, float]]:
        """
        使用Hybrid-A*算法搜索从起始节点到目标节点的路径。
        
        Hybrid-A*算法结合了离散A*和连续空间搜索，考虑车辆运动学约束，
        适用于非完整约束系统如汽车的路径规划。
        
        Args:
            start_node: 起始节点
            goal_node: 目标节点
            occupancy_map: 占用栅格地图
        
        Returns:
            list: 路径点列表，包含[(x1,y1,heading1), (x2,y2,heading2), ...]\n            或者在找不到路径时返回空列表 []
        """
        # 1. 定义碰撞检测器 无碰撞则返回True，否则返回False
        collision_checker_ = collision_checker.CollisionChecker()
        
        # 2. 初始化两个空字典：openList和closedList
        # openList: 存储待探索的节点的字典，键为节点的一维索引（CalIndex1DWithAngle），值为节点对象
        # 包含所有已生成但尚未扩展的节点，从中选择f值最小的节点进行扩展
        openList = {}
        
        # closedList: 存储已探索过的节点的字典，键为节点的一维索引（CalIndex1DWithAngle），值为节点对象
        # 包含已扩展过的节点，用于防止重复探索和路径重建（从目标回溯到起点）
        closedList = {}
        
        # 3. TODO: 使用A*生成启发式地图
        h_map = self.holonomic_heuristic_Astar(goal_node, occupancy_map)
        
        # 计算节点代价的函数
        def cal_cost(node: Node, goal_node: Node, h_map: dict, occupancy_map: occupancy_grid.OccupancyGrid) -> float:
            """
            计算节点的总代价（f值 = g值 + h值）。
            
            Args:
                node: 当前节点
                goal_node: 目标节点
                occupancy_map: 占用栅格地图
                h_map: 启发式地图
            
            Returns:
                float: 节点的总代价
            """
            h_cost = CalHCost(node, goal_node, h_map, occupancy_map)
            if h_cost == float('inf'): # 如果启发式不可达，总代价也设为无穷大
                return float('inf')
            return node.g_cost + h_cost
        
        # 4. 初始化起始节点
        map_width, map_height = occupancy_map.get_grid_size(), occupancy_map.get_grid_size()
        start_node.h_cost = CalHCost(start_node, goal_node, h_map, occupancy_map)
        start_node_id = start_node.CalIndex1DWithAngle(map_width, map_height)
        openList[start_node_id] = start_node

        # 5. 初始化优先队列，存储 (f_cost, node_id)
        pq = []
        heappush(pq, (start_node.h_cost, start_node_id)) # 初始 f_cost = g_cost(0) + h_cost

        # 添加最大迭代次数限制
        max_iterations = 10000  # 根据实际情况调整
        iteration_count = 0
        
        while pq and iteration_count < max_iterations:
            iteration_count += 1
            
            # 如果迭代次数过多，尝试使用Dubins shot
            if iteration_count > max_iterations * 0.8:  # 在接近最大迭代次数时
                is_shot, dubin_path = DubinShot(current_node, goal_node, occupancy_map)
                if is_shot:
                    print("Dubins shot successful after many iterations!")
                    final_path = self.get_final_path(closedList, dubin_path, current_node, goal_node)
                    break
                
            # 7. 从优先队列中取出f值最小的节点
            f_cost, current_node_id = heappop(pq)

            # 如果节点已在closedList中，跳过
            if current_node_id in closedList:
                continue

            # 将节点从openList移到closedList
            current_node = openList.pop(current_node_id)
            closedList[current_node_id] = current_node

            # 8. 尝试Dubins shot连接到目标
            is_shot, dubin_path = DubinShot(current_node, goal_node, occupancy_map)
            if is_shot:
                print("Dubins shot successful!")
                final_path = self.get_final_path(closedList, dubin_path, current_node, goal_node)
                break # 找到路径，退出循环

            # 9. 生成后继节点
            successors = GenerateSuccessors(current_node, current_node_id, goal_node)

            for successor in successors:
                # 10. 检查后继节点是否有效
                # - 边界检查
                if not occupancy_map.is_inside(successor.CalIndex2D()):
                    continue
                # - 碰撞检查
                successor_tuple = (successor.x, successor.y, successor.heading)
                if not collision_checker_.collision_check(successor_tuple, occupancy_map):
                    continue

                successor_id = successor.CalIndex1DWithAngle(map_width, map_height)

                # - 是否已在closedList中
                if successor_id in closedList:
                    continue

                # 11. 计算后继节点的代价
                successor.h_cost = CalHCost(successor, goal_node, h_map, occupancy_map)
                new_f_cost = successor.g_cost + successor.h_cost

                # 如果启发式代价为无穷大，则此路径不可行
                if new_f_cost == float('inf'):
                    continue

                # 12. 更新openList和优先队列
                if successor_id not in openList or new_f_cost < cal_cost(openList[successor_id], goal_node, h_map, occupancy_map):
                    openList[successor_id] = successor
                    heappush(pq, (new_f_cost, successor_id))

        if iteration_count >= max_iterations:
            print("Hybrid A* reached maximum iterations without finding a path.")
            return []

        return final_path

    def get_final_path(self, closed_set: Dict[int, Node], dubin_path: List[Any], current_node: Node, goal_node: Node) -> List[Tuple[float, float, float]]:
        """
        从closed_set和Dubins路径重建最终路径。\n\n        Args:\n            closed_set: 包含已探索节点的字典\n            dubin_path: Dubins路径点列表 [(x, y, heading), ...]\n            current_node: 连接到Dubins路径的最后一个Hybrid A*节点\n            goal_node: 目标节点\n\n        Returns:\n            list: 最终路径点列表 [(x, y, heading), ...]\n        """
        path = []
        # 添加Dubins路径部分 (注意格式)
        if dubin_path:
            for point in dubin_path:
                 # 确保添加的是 (x, y, heading) 元组
                 if len(point) >= 3:
                     path.append((point[0], point[1], normalize_angle(point[2])))
                 elif len(point) == 2: # 如果Dubins库只返回x,y，需要补充heading
                     # 这里假设Dubins路径的航向是连续变化的，最后一个点的航向接近目标航向
                     # 更好的做法是让Dubins库返回航向
                     # 暂时使用最后一个Hybrid A*节点的航向或目标节点的航向作为近似
                     path.append((point[0], point[1], goal_node.heading)) # 需要 goal_node 可访问

        # 回溯Hybrid A*部分
        node = current_node
        while node.predecssor is not None:
            # 插入到路径开头
            path.insert(0, (node.x, node.y, normalize_angle(node.heading)))
            node = closed_set[node.predecssor]
        # 添加起始节点
        path.insert(0, (node.x, node.y, normalize_angle(node.heading)))

        return path
    