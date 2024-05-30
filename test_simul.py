import subprocess
import re
import time

def run_command(task, load_run, checkpoint):
    command = f"python ee478_utils/tests/eval_success_rate.py --task={task} --load_run={load_run} --checkpoint={checkpoint}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout

def extract_success_rate(output):
    match = re.search(r'Success rate: tensor\(([\d.]+)', output)
    if match:
        return float(match.group(1))
    return None

def find_best_checkpoint(task, load_run, start_checkpoint, end_checkpoint, step):
    best_checkpoint = None
    best_success_rate = -1

    for checkpoint in range(start_checkpoint, end_checkpoint + 1, step):
        print(f"Running command for checkpoint: {checkpoint}")
        output = run_command(task, load_run, checkpoint)
        success_rate = extract_success_rate(output)

        if success_rate is not None and success_rate > best_success_rate:
            best_success_rate = success_rate
            best_checkpoint = checkpoint
        
        print(f"Checkpoint: {checkpoint}, Success rate: {success_rate}")
        # 잠시 대기 (옵션: 시스템 과부하 방지)
        time.sleep(1)

    return best_checkpoint, best_success_rate

if __name__ == "__main__":
    task = "ee_robotics"
    load_run = "/home/seunghyun/ee478/legged_gym/logs/ee_robotics/May21_18-31-31_"
    start_checkpoint = 1000
    end_checkpoint = 2000
    step = 50

    best_checkpoint, best_success_rate = find_best_checkpoint(task, load_run, start_checkpoint, end_checkpoint, step)

    print(f"\nBest Checkpoint: {best_checkpoint}, Best Success Rate: {best_success_rate}")

