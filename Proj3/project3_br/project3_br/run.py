import subprocess
import argparse

def run_task(task_index, debug=False):
    command = ['python3', './project3/project3_main.py', '--task-index', str(task_index)]
    if debug:
        command.append('-v')
        print("启用调试模式...")
    print(f"运行任务 {task_index}...")
    subprocess.run(command)

def run_all_tasks():
    for task_index in range(11):  # Range from 0 to 10
        run_task(task_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tasks')
    parser.add_argument('--task-index', type=int, help='Index of the task to run (0-10)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug information')
    args = parser.parse_args()

    if args.task_index is not None:
        if 0 <= args.task_index <= 10:
            run_task(args.task_index, args.verbose)
        else:
            print("Invalid task index. Please provide a value between 0 and 10.")
    else:
        run_all_tasks()
