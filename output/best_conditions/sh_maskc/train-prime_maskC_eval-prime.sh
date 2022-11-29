#! /usr/bin/bash
mkdir ../prime_maskC
cd ../prime_maskC
rsync -auv --include="*/" --include="*BL62LO_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank*" --exclude="*" --exclude="*BL62LO_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_prime_maskcysteine/ "./best_prime/" --progress
cd ./best_prime
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *BL62LO_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank-*
rmdir bootstrapping
rmdir raw
touch BL62LO_Mask_Peptide_EL_rank_mut_core_blsm_mut_score-core_mutation_score-ratio_rank