#! /usr/bin/bash
mkdir ../cedar_maskC
cd ../cedar_maskC
rsync -auv --include="*/" --include="*BL62LO_Mask_Peptide_trueHLA_EL_rank_dissimilarity_score-blsm_mut_score-mutation_score*" --exclude="*" --exclude="*BL62LO_Mask_Peptide_trueHLA_EL_rank_dissimilarity_score-blsm_mut_score-mutation_score-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_cedar_maskcysteine/ "./best_ibel/" --progress
cd ./best_ibel
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *BL62LO_Mask_Peptide_trueHLA_EL_rank_dissimilarity_score-blsm_mut_score-mutation_score-*
rmdir bootstrapping
rmdir raw
touch BL62LO_Mask_Peptide_trueHLA_EL_rank_dissimilarity_score-blsm_mut_score-mutation_score