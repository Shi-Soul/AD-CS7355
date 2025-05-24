import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

import occupancy_grid
import path_finding_algorithms

def load_test_case(file_path):
    """
    从文件加载测试用例
    
    Args:
        file_path: 测试用例文件路径
        
    Returns:
        tuple: (occupancy_map, start_point, goal_point)
    """
    with open(file_path, 'r') as f:
        test_case = json.load(f)
    
    # 重建占用栅格地图
    grid_size = len(test_case['occupancy_grid'])
    og_map = occupancy_grid.OccupancyGrid(
        grid_size, 
        test_case['resolution'], 
        tuple(test_case['center_position'])
    )
    og_map.occupancy_grid = np.array(test_case['occupancy_grid'], dtype=np.int8)
    
    # 获取起点和终点
    start_point = tuple(test_case['start_point'])
    goal_point = tuple(test_case['goal_point'])
    
    print(f"已加载测试用例: {file_path}")
    return og_map, start_point, goal_point

def main(test_case_path, task_index):
    """
    加载并运行指定的测试用例
    
    Args:
        test_case_path: 测试用例文件路径
    """
    # 加载测试用例
    og_map, start_point, goal_point = load_test_case(test_case_path)
    
    # 创建起点和终点节点
    start_node = path_finding_algorithms.Node(start_point[0], start_point[1], np.pi / 2.0)
    goal_node = path_finding_algorithms.Node(goal_point[0], goal_point[1], np.pi / 2.0)
    
    # 创建输出目录
    if not os.path.exists('./project3/task1_output'):
        os.makedirs('./project3/task1_output')
    
    # 可视化加载的地图
    plt.figure(figsize=(10, 10))
    plt.imshow(og_map.occupancy_grid.T, cmap='binary', origin='lower')
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='start point')
    plt.plot(goal_point[0], goal_point[1], 'ro', markersize=10, label='goal point')
    plt.grid(True)
    plt.legend()
    plt.title('Loaded Test Case Map')
    plt.savefig(f'./project3/task1_output/loaded_map{task_index}.png', dpi=300)
    
    # 使用路径规划算法寻找路径
    path_finder = path_finding_algorithms.PathFindingAlgorithm("Hybrid-A*")
    path = path_finder.search_path(start_node, goal_node, og_map, 1)
    
    # 将路径填充到地图并保存
    print("Path planning completed, saving results...")
    
    if path:
        # 创建包含路径的地图副本
        path_map = og_map.occupancy_grid.copy()
        og_map.occupancy_grid = path_map

        for point in path:
            og_map.set_data((int(point[0]), int(point[1])), 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(og_map.occupancy_grid.T, cmap='binary', origin='lower')
        plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='start point')
        plt.plot(goal_point[0], goal_point[1], 'ro', markersize=10, label='goal point')
        plt.grid(True)
        plt.legend()
        plt.title('Path for Loaded Test Case')
        plt.savefig(f'./project3/task1_output/test_case_path{task_index}.png', dpi=300)
    else:
        print("未能找到有效路径!")
    
    print("测试用例运行完成! 结果保存在 './project3/task1_output/' 目录中")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tasks')
    parser.add_argument('--task-index', type=int, help='Index of the task to run (0-4)')
    args = parser.parse_args()

    if args.task_index is not None:
        if 0 <= args.task_index <= 4:
            main(f'./project3/task1/test_case{args.task_index}.json', args.task_index)
        else:
            print("Invalid task index. Please provide a value between 0 and 4.")
    else:
        for i in range(5):
            main(f'./project3/task1/test_case{i}.json', i)