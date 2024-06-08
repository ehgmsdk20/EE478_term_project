#!/bin/bash

# Set the parameters
TASK="ee_robotics"
HEADLESS="--headless"
LOG_ROOT_DIR="/home/seunghyun/Downloads/ee478/legged_gym/logs/ee_robotics"
CHECKPOINT_INTERVAL=50
START_CHECKPOINT=1000
DEFAULT_MAX_ITERATIONS=5000

# Parse optional max_iterations parameter
MAX_ITERATIONS=${1:-$DEFAULT_MAX_ITERATIONS}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda init
conda activate ee478-env

sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Train the model
# python legged_gym/legged_gym/scripts/train.py --task=$TASK $HEADLESS --max_iterations=$MAX_ITERATIONS

# Get the latest log directory
LOG_DIR=$(ls -td ${LOG_ROOT_DIR}/* | head -1)

# Start roscore in the background
roscore &
ROSCORE_PID=$!
sleep 5  # Give roscore some time to start

# Create the CSV file
CSV_FILE="${LOG_DIR}/evaluation_results.csv"
echo "checkpoint,success_rate,velocity_tracking" > $CSV_FILE

# Evaluate the model from checkpoint 1000 to max iterations
for ((i=$START_CHECKPOINT; i<=$MAX_ITERATIONS; i+=$CHECKPOINT_INTERVAL)); do
  # Evaluate success rate
  SUCCESS_RATE_OUTPUT=$(python ee478_utils/ee478_utils/tests/eval_success_rate.py --task=$TASK --load_run=$LOG_DIR $HEADLESS --checkpoint=$i)
  SUCCESS_RATE=$(echo "$SUCCESS_RATE_OUTPUT" | grep "Success rate:" | awk -F'tensor\\(|,|\\)' '{print $2}' | sed 's/\.//')
  
  # Evaluate velocity tracking
  VELOCITY_TRACKING_OUTPUT=$(python ee478_utils/ee478_utils/tests/eval_velocity_tracking.py --task=$TASK --load_run=$LOG_DIR $HEADLESS --checkpoint=$i)
  VELOCITY_TRACKING=$(echo "$VELOCITY_TRACKING_OUTPUT" | grep "Velocity tracking error:" | awk -F'tensor\\(|,|\\)' '{print $2}')
  
  # Save the results to the CSV file
  echo "$i,$SUCCESS_RATE,$VELOCITY_TRACKING" >> $CSV_FILE
done

# Stop roscore
kill $ROSCORE_PID
sleep 3
