import math
import numpy as np
import matplotlib.pyplot as plt

def dist2d(point1, point2):
    """
    计算两点之间的欧几里得距离
    :param point1: 第一个点
    :param point2: 第二个点
    :return: 两点间的距离
    """
    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return math.sqrt(dist2)

def rotate_point(point, angle, offset_x, offset_y):
    """
    旋转点函数接受一个点(x, y)并按角度theta旋转。
    旋转是围绕原点(0, 0)进行的。函数返回一个新点
    (x', y')，该点旋转了theta弧度。
    
    :param point: 传入点的x和y坐标
    :param angle: 将点旋转特定角度
    :param offset_x: 将点的x坐标偏移特定量
    :param offset_y: 将点的y坐标偏移特定量
    :return: 包含两个值的元组(x, y)
    """
    x, y = point
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    x_rotated = x * cos_theta - y * sin_theta + offset_x
    y_rotated = x * sin_theta + y * cos_theta + offset_y
    return x_rotated, y_rotated

def global_2_local_coord_point(point, ego_position, ego_heading, resolution, grid_center_x, grid_center_y):
    """
    global_2_local_coord_point函数接受全局点、ego_position和ego_heading，
    网格的分辨率（以米为单位），以及网格的中心x和y坐标。然后返回
    该点的局部坐标版本（在占用地图中的位置）。
    
    :param point: 存储全局坐标系中点的x和y坐标
    :param ego_position: 确定全局坐标中自车的位置
    :param ego_heading: 旋转网格以与自车的航向对齐
    :param resolution: 将点从米转换为网格单位
    :param grid_center_x: 在x轴上居中网格
    :param grid_center_y: 在y轴上居中网格
    :return: 局部坐标系中的点
    """
    cos_theta = math.cos(ego_heading)
    sin_theta = math.sin(ego_heading)
    ego_position_x_grid = ego_position.x / resolution
    ego_position_y_grid = ego_position.y / resolution
    point_x_grid = point[0] / resolution
    point_y_grid = point[1] / resolution
    
    point_x = (point_x_grid - ego_position_x_grid) * sin_theta - (point_y_grid - ego_position_y_grid) * cos_theta + grid_center_x
    point_y = (point_x_grid - ego_position_x_grid) * cos_theta + (point_y_grid - ego_position_y_grid) * sin_theta + grid_center_y
    point_heading = point[2] + np.pi / 2.0 - ego_heading

    return (round(point_x), round(point_y), point_heading)

def global_2_local_coord_vertices(vertices, ego_position, ego_heading, resolution, grid_center_x, grid_center_y):
    """
    global_2_local_coord_vertices函数接受一个顶点列表、ego_position和heading，
    网格地图的分辨率（以米为单位），以及网格地图的中心x和y坐标。然后返回
    一个新列表，其中每个顶点都转换为局部坐标。
    
    :param vertices: 定义对象的形状
    :param ego_position: 获取世界中自车的位置
    :param ego_heading: 围绕ego_position旋转对象的顶点
    :param resolution: 将顶点从世界坐标转换为网格坐标
    :param grid_center_x: 在x轴上居中网格
    :param grid_center_y: 在y轴上居中网格
    :return: 元组列表
    """
    cos_theta = math.cos(ego_heading)
    sin_theta = math.sin(ego_heading)
    ego_position_x_grid = ego_position.x / resolution
    ego_position_y_grid = ego_position.y / resolution
    
    vertices_ = []
    for verticle in vertices:
        verticle_x = (verticle[0] / resolution - ego_position_x_grid) * sin_theta - (verticle[1] / resolution - ego_position_y_grid) * cos_theta + grid_center_x
        verticle_y = (verticle[0] / resolution - ego_position_x_grid) * cos_theta + (verticle[1] / resolution - ego_position_y_grid) * sin_theta + grid_center_y
        vertices_.append((verticle_x, verticle_y))
    return vertices_

def tmp_global_2_local_coord_vertices(vertices, ego_position, ego_heading, resolution, grid_center_x, grid_center_y):
    """
    global_2_local_coord_vertices函数接受一个顶点列表、ego_position和heading，
    网格地图的分辨率（以米为单位），以及网格地图的中心x和y坐标。然后返回
    一个新列表，其中每个顶点都转换为局部坐标。
    
    :param vertices: 定义对象的形状
    :param ego_position: 获取世界中自车的位置
    :param ego_heading: 围绕ego_position旋转对象的顶点
    :param resolution: 将顶点从世界坐标转换为网格坐标
    :param grid_center_x: 在x轴上居中网格
    :param grid_center_y: 在y轴上居中网格
    :return: 元组列表
    """
    cos_theta = math.cos(ego_heading)
    sin_theta = math.sin(ego_heading)
    ego_position_x_grid = ego_position[0] / resolution
    ego_position_y_grid = ego_position[1] / resolution
    
    vertices_ = []
    for verticle in vertices:
        verticle_x = (verticle[0] / resolution - ego_position_x_grid) * sin_theta - (verticle[1] / resolution - ego_position_y_grid) * cos_theta + grid_center_x
        verticle_y = (verticle[0] / resolution - ego_position_x_grid) * cos_theta + (verticle[1] / resolution - ego_position_y_grid) * sin_theta + grid_center_y
        vertices_.append((verticle_x, verticle_y))
    return vertices_

def local_2_global_coord_point(point, ego_position, ego_heading, resolution, grid_center_x, grid_center_y):
    """
    local_2_global_coord_point函数接受一个点(x, y)并将其从局部坐标系转换到全局坐标系。
    
    :param point: 获取相关点的x和y值
    :param ego_position: 计算ego_position_x和ego_position_y变量，用于计算每个点的全局坐标
    :param ego_heading: 旋转网格以匹配自车的航向
    :param resolution: 将网格坐标转换为真实世界坐标
    :param grid_center_x: 在x轴上居中网格
    :param grid_center_y: 在y轴上居中网格
    :return: 浮点数元组
    """
    cos_theta = math.cos(ego_heading)
    sin_theta = math.sin(ego_heading)
    ego_position_x_grid = ego_position.x / resolution
    ego_position_y_grid = ego_position.y / resolution

    point_x = ((point[0] - grid_center_x) * sin_theta + (point[1] - grid_center_y) * cos_theta + ego_position_x_grid) * resolution
    point_y = (-(point[0] - grid_center_x) * cos_theta + (point[1] - grid_center_y) * sin_theta + ego_position_y_grid) * resolution

    return (point_x, point_y)

def local_2_global_coord_vertices(vertices, ego_position, ego_heading, resolution, grid_center_x, grid_center_y):
    """
    local_2_global_coord_vertices函数接受一个顶点列表、ego_position和heading，
    网格地图的分辨率（以米为单位），以及网格地图的中心x-y坐标。然后返回
    一个包含所有从局部转换为全局坐标的顶点的列表。
    
    :param vertices: 存储多边形的顶点
    :param ego_position: 确定自车的位置
    :param ego_heading: 旋转局部坐标系中的顶点以与全局坐标系匹配
    :param resolution: 将网格坐标转换为真实世界坐标
    :param grid_center_x: 在x轴上居中网格
    :param grid_center_y: 在y轴上居中网格
    :return: 元组列表
    """
    cos_theta = math.cos(ego_heading)
    sin_theta = math.sin(ego_heading)
    ego_position_x_grid = ego_position.x / resolution
    ego_position_y_grid = ego_position.y / resolution
    
    vertices_ = []
    for verticle in vertices:
        verticle_x = ((verticle[0] - grid_center_x) * sin_theta + (verticle[1] - grid_center_y) * cos_theta + ego_position_x_grid) * resolution
        verticle_y = (-(verticle[0] - grid_center_x) * cos_theta + (verticle[1] - grid_center_y) * sin_theta + ego_position_y_grid) * resolution
        vertices_.append((verticle_x, verticle_y))
    return vertices_

def create_rotated_rectangle_in_place(center, width, height, angle):
    """
    create_rotated_rectangle_in_place函数接受一个中心点、宽度、高度和角度作为输入。
    然后创建四个点，这些点在每个方向上距离中心点半个宽度和高度。
    这些点围绕中心点按给定角度旋转，以创建具有任意旋转的矩形。
    
    :param center: 确定矩形的中心
    :param width: 确定矩形的宽度
    :param height: 设置矩形的高度
    :param angle: 旋转矩形
    :return: 四个点的列表
    """
    half_width = width / 2
    half_height = height / 2
    p1 = (-half_height, -half_width)
    p2 = (+half_height, -half_width)
    p3 = (+half_height, +half_width)
    p4 = (-half_height, +half_width)

    # 旋转点
    p1 = rotate_point(p1, angle, center.x, center.y)
    p2 = rotate_point(p2, angle, center.x, center.y)
    p3 = rotate_point(p3, angle, center.x, center.y)
    p4 = rotate_point(p4, angle, center.x, center.y)

    return [p1, p2, p3, p4]

def create_rotated_rectangle_offset_in_place(center, left_width, right_width, left_height, right_height, angle):
    """
    create_rotated_rectangle_offset_in_place函数根据中心、宽度、高度和角度创建一个旋转的矩形。
    
    :param center: 设置矩形的中心
    :param left_width: 确定矩形左侧的宽度
    :param right_width: 确定矩形右侧的宽度
    :param left_height: 确定矩形左侧的高度
    :param right_height: 确定矩形右侧的高度
    :param angle: 旋转矩形
    :return: 点列表
    """
    p1 = (-left_height, -left_width)
    p2 = (+right_height, -left_width)
    p3 = (+right_height, +right_width)
    p4 = (-left_height, +right_width)
    
    # 旋转点
    p1 = rotate_point(p1, angle, center.x, center.y)
    p2 = rotate_point(p2, angle, center.x, center.y)
    p3 = rotate_point(p3, angle, center.x, center.y)
    p4 = rotate_point(p4, angle, center.x, center.y)

    return [p1, p2, p3, p4]