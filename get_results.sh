#!/bin/bash

# Fixed dataset order for LaTeX table
datasets=("DUTS" "COME15K" "VT5000" "DIS5K" "COD10K" "SBU" "CDS2K" "ColonDB")

output=""

for dataset in "${datasets[@]}"; do
    # Get callback stats (input points)
    stats_file="is-sam2-right/base_sam_output/${dataset}/base_large/callback_stats.txt"
    if [ -f "$stats_file" ]; then
        inputs=$(grep "Average number of input points" "$stats_file" \
            | awk -F': ' '{print $2}' \
            | awk '{print $1}')
    else
        inputs="N/A"
    fi

    # Get final Average IoU
    log_file="results/base_${dataset}/infer_log.txt"
    if [ -f "$log_file" ]; then
        iou=$(grep "Average IoU:" "$log_file" | tail -n 1 | awk -F': ' '{print $2}')
    else
        iou="N/A"
    fi

    # Append to LaTeX-style line
    output+="${iou} & ${inputs} & "
done

# Trim trailing ampersand (&)
output=${output%"& "}

echo "$output"