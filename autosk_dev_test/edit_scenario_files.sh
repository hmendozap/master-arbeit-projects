#! /bin/bash
# Modify scenario files to make them compatible with fAnova implementation

for i in $( find . -name scenario.txt );
do
    echo "${i}"
    sed -i.bak '/cli-log-all-calls = false/d' ${i};
    sed -i.bak '/transform-crashed-quality-value = 2/d' ${i};
    sed -i.bak '/console-log-level = DEBUG/d' ${i};
    sed -i.bak '/log-level = DEBUG/d' ${i};
done

# Run this command to merge runs
# "~/autoskLearn_Setup/auto-sklearn/autosklearn/binaries/smac-v2.10.03/util/state-merge
# --directories 554_bac/state-run* --scenario-file 554_bac.scenario --outdir merged_runs_inv/"