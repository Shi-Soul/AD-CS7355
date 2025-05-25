import os
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import json

class OccupancyGrid:
    def __init__(self, grid_size, resolution, center_position):
        """
        创建网格地图
        :param grid_size: 单元格尺寸（以米为单位）
        :param resolution: 地图的分辨率
        :center_position: 地图的中心位置
        """
        self.grid_size = grid_size
        self.occupancy_grid = np.ones((grid_size, grid_size), dtype=np.int8)
        self.dim_cells = (grid_size, grid_size)
        self.resolution = resolution
        self.center_position = center_position
        
        # 标记已访问节点的2D数组（开始时，没有节点被访问过）
        # self.visited = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.visited_points = []
        self.visited_map = np.ones((grid_size, grid_size), dtype=np.int8)

    def update_visited_map(self):
        self.visited_map = self.occupancy_grid.copy()
        
    def set_visited_map(self, x_index, y_index):
        self.visited_map[round(x_index)][round(y_index)] = 1

    def add_visited_points(self, point):
        self.visited_points.append(point)

    def copy_from_map(self, map):
        self.occupancy_grid = map

    def get_resolution(self):
        return self.resolution
    
    def get_grid_size(self):
        return self.grid_size

    def reset_visited_map(self):
        """
        重置已访问地图。
        """
        self.visited = np.zeros(self.dim_cells, dtype=np.int8)

    def mark_grid_polygon(self, vertices, value = 1):
        """
        填充矩形占据的网格。
        :param vertices: 多边形顶点的列表
        """
        p = Polygon(vertices, closed = True)
        grid_x, grid_y = np.indices(self.occupancy_grid.shape)
        mask = p.contains_points(np.vstack((grid_x.flatten(), grid_y.flatten())).T, radius = 0.1)
        self.occupancy_grid[mask.reshape(self.occupancy_grid.shape)] = value

    def mark_visited(self, point_idx):
        """
        标记一个点为已访问。
        :param point_idx: 数据数组中的一个点 (x, y)
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('点超出地图边界')

        self.visited[x_index][y_index] = 1

    def is_visited(self, point):
        """
        检查给定点是否已被访问。
        :param point: 一个点 (x, y)
        :return: 如果给定点已被访问则返回True，否则返回False
        """
        x_index, y_index = point
        
        if self.visited[x_index][y_index] == 1:
            return True
        else:
            return False

    def get_data(self, point):
        """
        获取给定点的占用值。
        :param point: 一个点 (x, y)
        :return: a给定点的占用值
        """
        x_index, y_index = point
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('点超出地图边界')

        return self.occupancy_grid[x_index][y_index]

    def set_data(self, point, new_value):
        """
        设置给定点的占用值。
        :param point_idx: 一个点 (x, y)
        :param new_value: 新的占用值
        """
        x_index, y_index = point
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('点超出地图边界')

        self.occupancy_grid[x_index][y_index] = new_value

    def is_inside(self, point_idx):
        """
        检查给定点是否在地图内。
        :param point_idx: 一个点 (x, y)
        :return: 如果给定点在地图内则返回True，否则返回False
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            return False
        else:
            return True

    def is_occupied(self, point):
        """
        根据占用阈值检查给定点是否被占用。
        :param point: 一个点 (x, y)
        :return: 如果给定点被占用则返回True，否则返回False
        """
        x_index, y_index = point
        if self.get_data((x_index, y_index)) >= 0.5:
            return True
        else:
            return False

    def plot(self, search_count):
        """
        绘制网格地图
        """
        if not os.path.exists('./project3/figs'):
            os.makedirs('./project3/figs')
        fig, ax = plt.subplots()
        ax.imshow(self.occupancy_grid.T, cmap='binary', origin='lower', extent=(0, self.occupancy_grid.shape[1], self.occupancy_grid.shape[0], 0)) # 重新排序范围
        ax.grid(True, which='both', color='grey', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.show()
        plt.savefig(f'./project3/figs/{search_count}.png', dpi=300)

    def save_test_case(self, start_point, goal_point):
        """
        保存测试用例，包括地图信息、起点和终点
        
        Args:
            file_path: 保存文件的路径
            occupancy_map: 占用栅格地图
            start_point: 起点坐标 (x, y)
            goal_point: 终点坐标 (x, y)
        """
        # 创建保存目录
        file_path = './project3/tests/test_case.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备要保存的数据
        test_case = {
            'occupancy_grid': self.occupancy_grid.tolist(),
            'resolution': self.resolution,
            'center_position': self.center_position,
            'start_point': start_point,
            'goal_point': goal_point
        }
        
        # 保存到JSON文件
        with open(file_path, 'w') as f:
            json.dump(test_case, f)
        
        print(f"测试用例已保存到: {file_path}")