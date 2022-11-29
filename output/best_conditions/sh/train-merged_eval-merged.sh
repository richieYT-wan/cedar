#! /usr/bin/bash
mkdir ../merged
cd ../merged
rsync -auv --include="*/" --include="*onehot_Inverted-Mask_icore_mut_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank*" --exclude="*" --exclude="*onehot_Inverted-Mask_icore_mut_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_merged_fixed/ "./best_merged/" --progress
cd ./best_merged
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Inverted-Mask_icore_mut_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank-*
rmdir bootstrapping
rmdir raw
touch onehot_Inverted-Mask_icore_mut_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank