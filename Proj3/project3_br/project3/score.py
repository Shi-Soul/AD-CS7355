import os
import math

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def process_trajectory_file(filepath):
    """Read the trajectory.txt file and calculate the accumulated distance."""
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            if not lines:
                return 0
            # Extract positions from the file
            positions = []
            for line in lines:
                parts = line.strip().split(',')
                positions.append((float(parts[0]), float(parts[1])))
            # Calculate the accumulated distance
            accumulated_distance = 0
            for i in range(1, len(positions)):
                accumulated_distance += calculate_distance(positions[i-1], positions[i])
            return accumulated_distance
    except FileNotFoundError:
        return 0

def main():
    output_dir = './project3/task2_output'
    distances = {}

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        print("Output directory does not exist.")
        return

    # Iterate through each output_x directory
    for i in range(0, 11):  # Assuming output_1 to output_11
        folder_name = f'output_{i}'
        folder_path = os.path.join(output_dir, folder_name)
        trajectory_file = os.path.join(folder_path, 'trajectory.txt')
        
        if os.path.exists(folder_path):
            distance = process_trajectory_file(trajectory_file) + 2
        else:
            distance = 0
        
        distances[folder_name] = distance

    shortest_distance = [124.00, 124.00, 124.00, 124.00, 124.00, 93.86, 93.86, 93.86, 93.86, 93.86, 63.21]
    spl = []
    # Print the accumulated distances for each folder
    for i, folder in enumerate(distances):
        print(folder, "shortest distance:", shortest_distance[i], "actutal distance:", distances[folder])
        if distances[folder] == 0:
            spl.append(0.0)
        else:
            spl.append(shortest_distance[i] / (max(distances[folder], shortest_distance[i])))

    print(sum(spl), " ",len(spl), " ", sum(spl) / len(spl) )
    print("final score:", 15.0 if sum(spl) / len(spl) > 0.975 else 15.0 * sum(spl) / len(spl) / 0.975)

if __name__ == "__main__":
    main()
