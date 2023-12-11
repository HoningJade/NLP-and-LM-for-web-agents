#!/bin/bash

# Define the environment and parameters
result_dir="cache/91206_gpt35_16k_cot_na_deberta"
model="gpt-3.5-turbo-16k-0613"
instruction_path="agent/prompts/jsons/p_cot_id_actree_2s.json"
CONDA_ENV_NAME="webarena"

# Define server and other variables (fill in the placeholders with actual values)
SERVER="ec2-3-135-39-80.us-east-2.compute.amazonaws.com"
OPENAI_API_KEY=""
OPENAI_ORGANIZATION=""
# Define the environment variables to be used
ENV_VARIABLES="export SHOPPING='http://${SERVER}:7770';export SHOPPING_ADMIN='http://${SERVER}:7780/admin';export REDDIT='http://${SERVER}:9999';export GITLAB='http://${SERVER}:8023';export MAP='http://miniserver1875.asuscomm.com:3000';export WIKIPEDIA='http://${SERVER}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing';export HOMEPAGE='http://${SERVER}:4399';export OPENAI_API_KEY=${OPENAI_API_KEY};export OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION}"


# get the number of tmux panes
num_panes=$(tmux list-panes | wc -l)

# calculate how many panes need to be created
let "panes_to_create = 5 - num_panes"

# array of tmux commands to create each pane
tmux_commands=(
    'tmux split-window -h'
    'tmux split-window -v'
    'tmux select-pane -t 0; tmux split-window -v'
    'tmux split-window -v'
    'tmux select-pane -t 3; tmux split-window -v'
)

# create panes up to 5
for ((i=0; i<$panes_to_create; i++)); do
    eval ${tmux_commands[$i]}
done


# Read the task list from the file into an array
IFS=$'\n' read -d '' -r -a task_list < task_list.txt


# Function to run a job in a specific tmux pane
run_job() {
    tmux select-pane -t $1
    tmux send-keys "conda activate ${CONDA_ENV_NAME}; ${ENV_VARIABLES}; python browser_env/auto_login.py; until python run.py --test_start_idx $2 --test_end_idx $3 --model ${model} --instruction_path ${instruction_path} --result_dir ${result_dir}; do echo 'crashed' >&2; sleep 1; done" C-m
}

# Function to run a batch of jobs
run_batch() {
    local start_idx=0
    local batch_size=5
    local total_tasks=${#task_list[@]}
    local end_idx

    # Loop through the task list in batches
    for ((i=0; i<total_tasks; i+=batch_size)); do
        end_idx=$((i+batch_size-1))
        end_idx=$((end_idx<total_tasks ? end_idx : total_tasks-1))

        # Call run_job for each task ID
        for ((j=i; j<=end_idx; j++)); do
            run_job $((j-i+1)) ${task_list[j]} ${task_list[j+1]}
        done

        # Wait for all jobs to finish
        while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
            sleep 100  # wait for 10 seconds before checking again
        done

        # Check for errors and rerun if needed (use your own script for checking)
        # while ! python scripts/check_error_runs.py ${result_dir} --delete_errors --tolerance ${TOLERANCE}; do
        #     for ((j=i; j<=end_idx; j++)); do
        #         run_job $((j-i+1)) ${task_list[j]} ${task_list[j+1]}
        #     done
        #     # Wait again for all jobs to finish
        #     while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
        #         sleep 100  # wait for 10 seconds before checking again
        #     done
        # done
    done
}

run_batch