#!/bin/bash

# Run the jobs to plot distribution over validated configuration
datasets="38 46 179 184 293 389 554 772 917 1049 1111 1120 1128"
#experiments="GPU"
experiments="0305_benchmark"
#python -c 'import sys; print sys.real_prefix' 2>/dev/null && INVENV=1 || INVENV=0
for j in ${experiments};
do
    for i in ${datasets};
    do
        python validation_distro_plots.py /mhome/mendozah/autonet_${j}/results/experiment/ ${i}_bac /mhome/mendozah/autonet_${j}/results/plots_distributions/all/;
    done;
done;
exit 0
