#! /usr/bin/bash
mkdir ../prime_maskC
cd ../prime_maskC
rsync -auv --include="*/" --include="*BL62LO_KL_Peptide_trueHLA_EL_rank_dissimilarity_score-core_blsm_mut_score*" --exclude="*" --exclude="*BL62LO_KL_Peptide_trueHLA_EL_rank_dissimilarity_score-core_blsm_mut_score-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_prime_maskcysteine/ "./best_ibel/" --progress
cd ./best_ibel
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *BL62LO_KL_Peptide_trueHLA_EL_rank_dissimilarity_score-core_blsm_mut_score-*
rmdir bootstrapping
rmdir raw
touch BL62LO_KL_Peptide_trueHLA_EL_rank_dissimilarity_score-core_blsm_mut_score