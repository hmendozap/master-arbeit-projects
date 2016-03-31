#! /bin/bash
# Modify scenario files to make them compatible with fAnova implementation

for i in $( find . -name scenario.txt );
do
    echo "${i}"
    sed -i.bak '/cli-log-all-calls = false/d' ${i};
    sed -i '/transform-crashed-quality-value = 2/d' ${i};
    sed -i '/console-log-level = DEBUG/d' ${i};
    sed -i '/log-level = DEBUG/d' ${i};
done

smac_merge=$( /home/mendozah/autoskLearn_Setup/auto-sklearn/autosklearn/binaries/smac-v2.10.03/util/state-merge )
dataset_name=("MNIST" "OVA_Breast" "Splice")
dataset_code=("554" "1128" "46")
for i in {0..2};
do
    echo ${dataset_name[${i}]}
    output_dir=${dataset_name[${i}]}_merged_runs
    $( mkdir "${output_dir}" )
    ${smac_merge} --directories ${dataset_name[${i}]}/*/${dataset_code[${i}]}_bac/state-run* --scenario-file ${dataset_code[${i}]}_bac.scenario --outdir ${output_dir}
done

# Run this command to merge runs
# "~/autoskLearn_Setup/auto-sklearn/autosklearn/binaries/smac-v2.10.03/util/state-merge
# --directories 554_bac/state-run* --scenario-file 554_bac.scenario --outdir merged_runs_inv/"