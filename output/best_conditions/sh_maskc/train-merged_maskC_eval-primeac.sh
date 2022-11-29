#! /usr/bin/bash
mkdir ../merged_maskC
cd ../merged_maskC
rsync -auv --include="*/" --include="*onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-core_mutation_score*" --exclude="*" --exclude="*onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-core_mutation_score-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_merged_maskcysteine/ "./best_primeac/" --progress
cd ./best_primeac
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-core_mutation_score-*
rmdir bootstrapping
rmdir raw
touch onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-core_mutation_score