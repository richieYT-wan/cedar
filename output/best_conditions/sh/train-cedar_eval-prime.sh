#! /usr/bin/bash
mkdir ../cedar
cd ../cedar
rsync -auv --include="*/" --include="*onehot_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score*" --exclude="*" --exclude="*onehot_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_cedar_fixed/ "./best_prime/" --progress
cd ./best_prime
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-*
rmdir bootstrapping
rmdir raw
touch onehot_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score