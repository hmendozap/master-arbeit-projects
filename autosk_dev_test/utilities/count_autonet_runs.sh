#!/bin/bash
# Small script to count runs over autonet experiments

datasets="38 46 179 184 554 772 917 1049 1111 1120 1128 293 389"
a_dir="/mhome/mendozah/autonet_0305_benchmark/results/experiment/"
b_dir="/mhome/mendozah/autonet_full/results/experiment/"
for i in $datasets;
do
    a=$(find ${a_dir}${i}_bac/DeepFeedNet/ -name "num_run_*" | xargs awk '{SUM += $1} END {print SUM}')
    b=$(find ${b_dir}${i}_bac/DeepNetIterative/ -name "num_run_*" | xargs awk '{SUM += $1} END {print SUM}')
    echo "${i} dataset"
    echo "scale=2; $b/$a" | bc
done
