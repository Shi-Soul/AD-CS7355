from math_utils import *
import occupancy_grid
import math

"""
TODO: 请自定义你自己的圆形偏移量和圆半径
      1. 圆形偏移量：从车辆中心到特定圆的中心的距离，例如 [-2.0, 0.0, 2.0]
      2. 圆半径：用于检测碰撞的圆的半径，例如 2.0
      自车的尺寸为vehicle length: 3.705369472503662  width: 1.788678526878357
"""
class CollisionChecker:
    """
    碰撞检测器类，假设车辆由多个圆组成来进行碰撞检测。
    
    通过将车辆简化为沿车身轴线分布的多个圆，可以高效地进行碰撞检测。
    圆的位置由circle_offsets参数确定，圆的半径由circle_radii参数确定。
    
    Attributes:
        _circle_offsets: 圆心相对于车辆中心的偏移距离列表
        _search_boundary: 碰撞检测的搜索边界
    """
    def __init__(self, circle_offsets = [-2.0, 0.0, 2.0], search_boundary = 5.0):
        """
        初始化碰撞检测器。
        
        Args:
            circle_offsets: 圆心相对于车辆中心的偏移距离列表，单位为米
            search_boundary: 碰撞检测的搜索边界，单位为米
        """
        self._circle_offsets = circle_offsets
        self._search_boundary = search_boundary

    def collision_check(self, point, occupancy_grid, circle_radii = 2.0):
      """
      检查指定点在占用栅格地图中是否无碰撞。
      
      此函数使用圆模型检测车辆在给定位置是否与障碍物发生碰撞。
      车辆被简化为沿车身轴线分布的多个圆，如果任一圆与障碍物重叠，则认为发生碰撞。
      
      Args:
          point: 待检测点，格式为 (x, y, heading)
          occupancy_grid: 占用栅格地图
          circle_radii: 碰撞检测圆的半径，单位为米
      
      Returns:
          bool: 注意！！！如果无碰撞则返回True，否则返回False
      """
      point_x, point_y, point_heading = point
      resolution = occupancy_grid.get_resolution()
      search_boundary_in_map = round(self._search_boundary / resolution)
      circle_locations = [(point_x + i / resolution * math.cos(point_heading), point_y + i / resolution * math.sin(point_heading)) for i in self._circle_offsets]
      
      for i in range(-search_boundary_in_map, search_boundary_in_map, 2):
          for j in range(-search_boundary_in_map, search_boundary_in_map, 2):
              new_point = (round(point_x + i), round(point_y + j))
              # 检查新位置是否在地图内
              if not occupancy_grid.is_inside(new_point):
                  continue
              
              # 检查新位置是否是障碍物
              if not occupancy_grid.is_occupied(new_point):
                  continue
              
              for offset_point in circle_locations:
                  if dist2d(offset_point, new_point) < circle_radii / resolution:
                      return False
              
      return True