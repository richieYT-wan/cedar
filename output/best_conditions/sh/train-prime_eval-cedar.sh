#! /usr/bin/bash
mkdir ../prime
cd ../prime
rsync -auv --include="*/" --include="*onehot_Mask_icore_mut_EL_rank_mut_dissimilarity_score-blsm_mut_score*" --exclude="*" --exclude="*onehot_Mask_icore_mut_EL_rank_mut_dissimilarity_score-blsm_mut_score-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_prime_fixed/ "./best_cedar/" --progress
cd ./best_cedar
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Mask_icore_mut_EL_rank_mut_dissimilarity_score-blsm_mut_score-*
rmdir bootstrapping
rmdir raw
touch onehot_Mask_icore_mut_EL_rank_mut_dissimilarity_score-blsm_mut_score