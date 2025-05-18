import dubins
from heapq import heappush, heappop
import occupancy_grid
import collision_checker
import math_utils
import math
import numpy as np
import time
from typing import Tuple, List, Dict, Any

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
        min_heading = round(-math.pi / YAW_GRID_RESOLUTION) - 1
        return (round(self.heading / YAW_GRID_RESOLUTION) - min_heading) * width * height \
                + round(self.y) * width + round(self.x)

    def CalIndex1D(self, width, height):
        return round(self.y) * width + round(self.x)

    def CalIndex2D(self):
        return (round(self.x), round(self.y))

def GenerateSuccessors(node: Node, node_id: int) -> List[Node]:
    """
    生成后继节点列表，应用车辆运动学模型产生有效的后继状态。
    
    Args:
        node: 当前节点，包含位置和航向信息
        node_id: 当前节点的唯一标识符
    
    Returns:
        list: 可能的后继节点列表
    """
    successors = []
    
    # TODO: 定义转向空间
    dela_thetas = [] # 角度值，例如 [-30, -15, 0, 15, 30] 角度制 注意转换为弧度制，同时delta_theta需要为YAW_GRID_RESOLUTION的倍数
    
    for delta_theta in dela_thetas:
        # TODO: 生成后继节点
        distance = 5.0 # 移动距离
        # 1. 计算新的位置和航向(new_x, new_y, new_theta)
        
        # 2. 计算新节点的g_cost（new_g_cost根据路径长度，假如delta_theta不为0，则需要乘以一个转弯的惩罚系数如1.5）
        
        # 3. 创建新的后继节点
        successors.append(Node(new_x, new_y, new_theta, new_g_cost, node.h_cost, node_id))
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
    # 1. 使用h_map查找起始节点位置的代价
    # 2. 如果节点不在h_map中，返回适当的默认值
    map_width, map_height = occupancy_map.get_grid_size(), occupancy_map.get_grid_size()
    node_idx_1d = start_node.CalIndex1D(map_width, map_height)
    if node_idx_1d not in h_map:
        return 999999.0
    return h_map[node_idx_1d]

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
    dubins_path = dubins.shortest_path((start_node.x, start_node.y, start_node.heading),
                                       (goal_node.x, goal_node.y, goal_node.heading), 
                                       1.0)
    dubins_cost = dubins_path.path_length()
    return dubins_cost
    
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
    astar_cost = CalAStarPathCost(start_node, h_map, occupancy_map)
    dubin_cost = CalDubinPathCost(start_node, goal_node)
    return max(astar_cost, dubin_cost)

def DubinShot(start_node: Node, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid) -> Tuple[bool, List[Any]]:
    #TODO: path = dubins.shortest_path(q0, q1, turning_radius), 你可以通过调节第三个参数控制转弯半径
    dubins_path = dubins.shortest_path((start_node.x, start_node.y, start_node.heading),
                                       (goal_node.x, goal_node.y, goal_node.heading), 
                                       1.0)
    points, _ = dubins_path.sample_many(2.0)
    collision_checker_ = collision_checker.CollisionChecker()
    for point in points:
        if not collision_checker_.collision_check(point, occupancy_map):
            return False, None
    return True, points

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
            return self.HybridAStar(start_node, goal_node, occupancy_map)
        
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
        
        goal_node_idx_1d = goal_node.CalIndex1D(map_width, map_height)
        openset[goal_node_idx_1d] = goal_node
        
        # pq: 优先队列，按f值排序的节点列表，用于高效取出f值最小的节点
        # 存储元组 (f值, 节点的一维索引)，f值越小优先级越高
        pq = []
        pq.append((0, goal_node_idx_1d))
        
        while 1:
            # TODO: 在占用图中扩展所有节点并将其g_cost存储在closedset中
            # 1. 从优先队列pq中取出f值最小的节点
            # 2. 如果队列为空，返回closedset
            # 3. 将节点从openset移到closedset
            # 4. 遍历所有可能的移动方向
            #    - 跳过障碍物 (occupancy_map.is_occupied(node.CalIndex2D()))
            #    - 跳过超出occupancy map边界的节点 (not occupancy_map.is_inside(node.CalIndex2D()))
            #    - 跳过已拓展过的节点
            # 5. 计算新节点的代价并更新openset和pq
            pass
        
        return closedset
        
    def HybridAStar(self, start_node: Node, goal_node: Node, occupancy_map: occupancy_grid.OccupancyGrid) -> List[Tuple[float, float, float]]:
        """
        使用Hybrid-A*算法搜索从起始节点到目标节点的路径。
        
        Hybrid-A*算法结合了离散A*和连续空间搜索，考虑车辆运动学约束，
        适用于非完整约束系统如汽车的路径规划。
        
        Args:
            start_node: 起始节点
            goal_node: 目标节点
            occupancy_map: 占用栅格地图
        
        Returns:
            list: 路径点列表，包含[(x1,y1,heading1), (x2,y2,heading2), ...]
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
            return node.g_cost + CalHCost(node, goal_node, h_map, occupancy_map)
        
        # 4. 初始化起始节点
        map_width, map_height = occupancy_map.get_grid_size(), occupancy_map.get_grid_size()
        
        # pq: 优先队列，按f值排序的节点列表，用于高效取出f值最小的节点
        # 存储元组 (f值, 节点的一维索引)，f值越小优先级越高
        pq = [] 
        
        start_node_id = start_node.CalIndex1DWithAngle(map_width, map_height)
        openList[start_node_id] = start_node
        heappush(pq, (cal_cost(start_node, goal_node, h_map), start_node_id))

        current = None
        c_id = None
        dubin_path = []

        while True:
            # 5. TODO: 如果openList为空，返回空路径
            # 如果openList为空，表示无法找到路径，返回空列表(return [])
            
            # 6. TODO: 获取f值最小的下一个待扩展节点
            # 从优先队列中取出f值最小的节点作为当前节点
            
            # 7. 检查当前节点是否可以直接连接到目标节点（通过dubinshot或其他算法）
            # 检查当前节点是否可以直接连接到目标节点
            success, dubin_path = DubinShot(current, goal_node, occupancy_map)  # 使用Dubins曲线尝试连接
            if success:
                break
            
            # 8. TODO: 生成后继节点
            for successor in GenerateSuccessors(current, c_id):
                successor_idx_1d = successor.CalIndex1DWithAngle(map_width, map_height)

                # 9. TODO: 如果后继节点已经在闭集中，跳过它
                # 如果后继节点已经在闭集中，跳过
                    
                # 10. TODO: 如果后继节点与障碍物碰撞，跳过它
                # 使用collision_checker检查后继节点是否与障碍物碰撞
                
                # 11. TODO: 如果找到更好的路径或节点不在开集中，则将后继节点加入开集
                # 如果找到更好的路径或节点不在开集中，更新/添加节点到开集与队列中
        
        # 12. TODO: 定义从起始节点到目标节点获取最终路径的函数
        def get_final_path(closed_set: Dict[int, Node], dubin_path: List[Any], current_node: Node) -> List[Tuple[float, float, float]]:
            """
            从搜索结果中重建完整路径。
            
            Args:
                closed_set: 闭集
                dubin_path: Dubins路径（如果找到）
                current_node: 当前节点（终点或连接点）
            
            Returns:
                list: 完整路径点列表
            """
            path = []
            
            # 13. TODO: 从目标节点回溯到起始节点
            # 从当前节点回溯到起始节点
            pre_node_id = current_node.predecssor
            while pre_node_id is not None:
                # 获取前驱节点并添加到路径中
                # 更新pre_node_id为前驱节点的前驱
                pass
            
            # 14. TODO: 反转路径使其从起点开始
            # 反转路径使其从起点开始

            # 15. 添加Dubins路径
            path += [(point[0], point[1], point[2]) for point in dubin_path]

            return path
        
        # 获取最终路径并返回
        path = get_final_path(closedList, dubin_path, current)
        return path
    