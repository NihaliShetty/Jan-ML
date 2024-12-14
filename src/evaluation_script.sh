#!/bin/bash

# Define possible values for each parameter
tasks=("mc" "da")
viper_values=(true false)
rationale_values=(true false)

# Loop through all possible combinations of task, viper, and rationale
for task in "${tasks[@]}"; do
    for viper in "${viper_values[@]}"; do
        for rationale in "${rationale_values[@]}"; do
            # Build the command dynamically
            cmd="python evaluate.py --task $task"
            
            if [ "$viper" = "true" ]; then
                cmd+=" --viper"
            fi

            if [ "$rationale" = "true" ]; then
                cmd+=" --rationale"
            fi

            echo "Running: $cmd"
            # Execute the command
            $cmd
        done
    done
done
