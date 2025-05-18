import numpy as np
from scipy import interpolate
import math

class PathOptimizer:
    def __init__(self):
        pass

    def sample_path(self, path):
        if len(path) < 3:
            return path
        
        sampled_path = []
        for index in range(0, len(path), 3):
            sampled_path.append(path[index])
        return sampled_path  

    def smooth_path_spline(self, path):
        if len(path) < 3:
            return path

        # 将路径转换为numpy数组以便更容易操作
        path_array = np.array(path)
        #print(path_array)
        
        tck,u = interpolate.splprep([path_array[:, 0], path_array[:, 1]], k=3, s=32)

        u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u, tck)
        
        # 计算每个点的切向量
        dx, dy = interpolate.splev(u, tck, der=1)
        tangent_vectors = np.array([dx, dy]).T

        # 归一化切向量以获得航向方向
        normalized_tangent_vectors = tangent_vectors / np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]

        # 将x、y坐标和航向方向组合成平滑路径
        smoothed_path = [(x, y, np.arctan2(dy, dx)) for x, y, dx, dy in zip(out[0], out[1], normalized_tangent_vectors[:, 0], normalized_tangent_vectors[:, 1])]

        return smoothed_path