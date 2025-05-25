# CS7355 作业 3 - Task 1 实现总结 (path_finding_algorithms.py)

# A* (A-Star) 寻路算法简介

A\* 算法是一种在图形或网格中寻找从起点到终点最短路径的常用**启发式搜索算法**。它结合了 Dijkstra 算法（查找最短路径）和贪心最佳优先搜索（使用启发式信息快速导向目标）的优点。

## 核心思想

A\* 算法的核心思想是通过评估每个节点的**代价函数 `f(n)`** 来决定下一个要探索的节点。这个代价函数结合了两个部分：

1.  **`g(n)`**: 从**起点**到当前节点 `n` 的**实际代价**（已经走过的路径长度）。
2.  **`h(n)`**: 从当前节点 `n` 到**终点**的**估计代价**（启发式函数，Heuristic Function）。这是一个基于问题特性的“猜测”值，用于估计剩余路径的成本。

代价函数公式为：
\[ f(n) = g(n) + h(n) \]

算法的目标是在每次迭代中，选择具有**最低 `f(n)` 值**的节点进行扩展，因为它被认为是通往目标的最有希望的路径上的节点。

## 关键组件

*   **节点 (Node)**: 图或网格中的一个位置。
*   **起点 (Start Node)**: 路径的开始位置。
*   **终点 (Goal Node)**: 路径的目标位置。
*   **`g(n)`**: 从起点到节点 `n` 的实际移动代价。
*   **`h(n)`**: 从节点 `n` 到终点的估计移动代价（启发式）。
*   **`f(n)`**: 节点 `n` 的总评估代价 (`g(n) + h(n)`)。
*   **开放列表 (Open Set/Open List)**: 存储已发现但尚未完全评估的节点。通常使用优先队列实现，按 `f(n)` 值排序。
*   **关闭列表 (Closed Set/Closed List)**: 存储已经完成评估的节点，避免重复处理。

## 算法步骤

1.  **初始化**:
    *   创建一个空的开放列表和关闭列表。
    *   将**起点**添加到开放列表中。设置起点的 `g` 值为 0，计算其 `h` 值和 `f` 值。

2.  **循环**: 当开放列表不为空时：
    *   **选择节点**: 从开放列表中选择（并移除）`f(n)` 值最低的节点，记为**当前节点 `n`**。
    *   **检查目标**: 如果当前节点 `n` 是**终点**，则路径找到。可以通过回溯每个节点的父节点来重建完整路径，然后算法结束。
    *   **移至关闭列表**: 将当前节点 `n` 添加到关闭列表中。
    *   **遍历邻居**: 对于当前节点 `n` 的每一个**邻居节点 `m`**：
        *   **跳过已处理**: 如果邻居 `m` 已经在关闭列表中，则忽略它。
        *   **计算代价**: 计算从起点经过当前节点 `n` 到达邻居 `m` 的实际代价（`tentative_g = g(n) + cost(n, m)`）。
        *   **发现新节点或找到更优路径**:
            *   如果邻居 `m` **不在**开放列表中，或者新计算的 `tentative_g` **小于** `m` 当前记录的 `g` 值：
                *   设置 `m` 的父节点为 `n`。
                *   更新 `m` 的 `g` 值为 `tentative_g`。
                *   计算并更新 `m` 的 `h` 值和 `f` 值。
                *   如果 `m` 不在开放列表中，则将其加入开放列表。

3.  **失败**: 如果开放列表变为空，但仍未找到终点，则表示从起点到终点不存在路径。

## 启发式函数 (`h(n)`)

启发式函数的选择对 A\* 算法的性能至关重要：

*   **可接受性 (Admissibility)**: `h(n)` 必须**从不**高估从 `n` 到终点的实际最短路径代价。这是保证 A\* 找到最优路径的关键。
*   **一致性 (Consistency)**: 对于任意节点 `n` 和其邻居 `m`，`h(n)` 必须小于等于 `cost(n, m) + h(m)`。一致性保证了可接受性，并且可以提高效率（避免重新打开已关闭的节点）。
*   **常见例子**:
    *   **曼哈顿距离 (Manhattan Distance)**: 在网格中，只能水平或垂直移动时使用。`h = |n.x - goal.x| + |n.y - goal.y|`。
    *   **欧几里得距离 (Euclidean Distance)**: 允许直线移动时的直线距离。`h = sqrt((n.x - goal.x)^2 + (n.y - goal.y)^2)`。

## 优点

*   **最优性**: 如果使用的启发式函数 `h(n)` 是可接受的（并且图的边代价非负），A\* 保证能找到最短路径。
*   **效率**: 通常比 Dijkstra 算法更高效，因为它利用启发式信息优先探索朝向目标的路径。

## 应用场景

A\* 算法广泛应用于各种需要寻路的问题中，例如：

*   视频游戏中的角色寻路。
*   机器人路径规划。
*   地图应用中的路线导航。
*   网络路由。

**目标:** 完成 `path_finding_algorithms.py` 脚本中的 Hybrid A* 算法实现，用于在给定的栅格地图上规划路径，满足 Task 1 的评估要求。关键约束是**仅修改此文件**。

## 主要实现内容 (Hybrid A*)

根据 `README.md` 的要求和代码中的 `TODO` 注释，我们主要完成了以下几个部分：

1.  **启发式函数 (`holonomic_heuristic_Astar`)**:
    *   **目的**: 为 Hybrid A* 提供启发式估计代价 (h_cost)，即从当前点到目标点的大致代价。此函数忽略车辆的朝向约束（全向移动），使用传统的 A* (或 Dijkstra) 算法。
    *   **实现**: 我们实现了从 **目标节点** 反向搜索的 A* 算法。算法维护一个开集 (`openset`) 和一个闭集 (`closedset`)，以及一个优先队列 (`pq`)。从目标节点开始（代价为0），不断扩展代价最小的节点，计算其邻居节点的代价，直到优先队列为空。最终 `closedset` 存储了地图上所有可达点到目标点的最小栅格距离代价（g_cost），这个代价就被用作 Hybrid A* 的启发式值 h。

2.  **后继节点生成 (`GenerateSuccessors`)**:
    *   **目的**: 模拟车辆从当前状态（位置 `x, y`，朝向 `heading`）出发，根据可能的转向操作，生成一系列可能的下一状态（后继节点）。
    *   **实现**:
        *   定义了一组离散的转向角度 `delta_thetas` (例如 -30°, -15°, 0°, 15°, 30°，转换为弧度)。
        *   对于每个转向角，假设车辆向前行驶固定距离 `distance`。
        *   计算新的朝向 `new_heading` (使用本地定义的 `normalize_angle` 归一化)。
        *   计算新的位置 `(new_x, new_y)`。
        *   计算移动代价 `new_g_cost`：基础代价是行驶距离 `distance`，如果发生了转向 (即 `delta_theta` 不为 0)，则乘以一个转弯惩罚系数 `turn_penalty` (例如 1.5)，以鼓励直线行驶。
        *   创建新的 `Node` 对象并添加到后继列表中。

3.  **Hybrid A* 主算法 (`HybridAStar`)**:
    *   **目的**: 结合考虑车辆运动学约束的搜索（通过 `GenerateSuccessors`）和启发式引导（通过 `holonomic_heuristic_Astar`），找到从起点到终点的可行且考虑朝向的路径。
    *   **实现**:
        *   **初始化**: 创建开集 (`openList`)、闭集 (`closedList`) 和优先队列 (`pq`)。调用 `holonomic_heuristic_Astar` 生成启发式地图 `h_map`。初始化起点，计算其 h_cost (使用 `CalHCost`)，并将其加入 `openList` 和 `pq`。
        *   **主循环**: 当优先队列 `pq` 不为空时循环执行：
            *   从 `pq` 中取出 f 值（f = g + h）最小的节点 `current_node`。
            *   将 `current_node` 从 `openList` 移到 `closedList`。
            *   **Dubins Shot**: 尝试直接使用 Dubins 曲线（一种计算两个带朝向的点之间最短路径的几何方法）连接 `current_node` 和 `goal_node`。调用 `DubinShot` 函数，该函数内部使用 `dubins.shortest_path` 计算路径，并采样路径上的点，使用 `collision_checker.collision_check` 检查这些点是否与障碍物碰撞。如果 Dubins 路径无碰撞，则路径查找成功，调用 `get_final_path` 重建路径并退出循环。
            *   **扩展节点**: 如果 Dubins Shot 不成功，则调用 `GenerateSuccessors` 生成 `current_node` 的所有后继节点 `successor`。
            *   **验证后继节点**: 对每个 `successor`：
                *   进行碰撞检测 (`collision_checker.collision_check`)。
                *   检查是否已在 `closedList` 中。
                *   如果有效，计算其 f_cost (`successor.g_cost + CalHCost(...)`)。
                *   如果该 `successor` 不在 `openList` 中，或者新的 f_cost 比 `openList` 中已有的更低，则将其加入/更新到 `openList` 和 `pq`。
        *   **路径重建 (`get_final_path`)**: 如果循环结束（Dubins Shot 成功），则从连接点 (`connect_node`，即 Dubins Shot 起始的那个节点) 开始，通过 `predecssor` 指针在 `closedList` 中回溯，直到起点，构建 Hybrid A* 部分的路径。然后将这部分与 Dubins 路径拼接起来。

4.  **辅助函数**:
    *   `CalAStarPathCost`: 从 `h_map` 中查找节点的 A* 启发式代价。
    *   `CalDubinPathCost`: 计算两节点间的 Dubins 曲线长度。
    *   `CalHCost`: 计算最终的启发式代价，通常取 `CalAStarPathCost` 和 `CalDubinPathCost` 的最大值，以提供更准确的估计。处理了代价为无穷大的情况。
    *   `DubinShot`: 检查两节点间是否存在无碰撞的 Dubins 路径。
    *   `normalize_angle` (本地添加): 将角度归一化到 `[-pi, pi)` 区间。

## 遇到的问题及解决方法

在实现和测试过程中，我们遇到了以下几个错误：

1.  **`AttributeError: module 'path_finding_algorithms' has no attribute 'Node'`**
    *   **问题**: 评估脚本无法从模块中找到 `Node` 类，尽管它已定义。同时发现 `get_final_path` 无法访问 `goal_node`。
    *   **解决**:
        *   修正了 `HybridAStar` 中对 `get_final_path` 的调用，传递了 `goal_node` 参数，并更新了 `get_final_path` 的函数签名。
        *   重新应用了整个文件的代码，并进行了一些小的重构和健壮性改进（如索引边界检查、添加循环退出条件等），确保代码的整体一致性。这通常能间接解决一些难以追踪的导入或环境问题。

2.  **`AttributeError: module 'math_utils' has no attribute 'normalize_angle'`**
    *   **问题**: 代码中调用了 `math_utils.normalize_angle`，但该函数在 `math_utils` 模块中不存在，或者我们不允许修改 `math_utils.py`。
    *   **解决**: 遵循**仅修改 `path_finding_algorithms.py`** 的原则，我们在该文件顶部定义了一个本地的 `normalize_angle` 函数，并替换了所有 `math_utils.normalize_angle` 的调用。

3.  **`TypeError: cannot unpack non-iterable Node object` (在 `collision_checker.py` 中)**
    *   **问题**: `collision_checker.py` 中的 `collision_check` 函数期望接收一个包含 `(x, y, heading)` 的元组，但我们从 `path_finding_algorithms.py` 中传递了 `Node` 对象。
    *   **解决**: 修改了 `path_finding_algorithms.py` 中所有调用 `collision_checker_.collision_check` 的地方（包括起点、终点、Dubins 路径点、后继节点的检查），确保传递的是形如 `(node.x, node.y, node.heading)` 的元组，而不是 `Node` 对象本身。

## 总结

通过实现上述核心功能并解决遇到的错误，我们成功完成了 `path_finding_algorithms.py` 的修改，使其能够通过 Task 1 的评估。关键在于理解 Hybrid A* 的流程、正确实现各组成部分（特别是启发式、后继生成和 Dubins Shot），并细心处理不同模块间的数据类型匹配问题。