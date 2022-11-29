#! /usr/bin/bash
mkdir ../prime_maskC
cd ../prime_maskC
rsync -auv --include="*/" --include="*onehot_Mask_Peptide_EL_rank_mut_blsm_mut_score-mutation_score-ratio_rank*" --exclude="*" --exclude="*onehot_Mask_Peptide_EL_rank_mut_blsm_mut_score-mutation_score-ratio_rank-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_prime_maskcysteine/ "./best_agg/" --progress
cd ./best_agg
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Mask_Peptide_EL_rank_mut_blsm_mut_score-mutation_score-ratio_rank-*
rmdir bootstrapping
rmdir raw
touch onehot_Mask_Peptide_EL_rank_mut_blsm_mut_score-mutation_score-ratio_rank