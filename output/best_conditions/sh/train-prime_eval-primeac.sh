#! /usr/bin/bash
mkdir ../prime
cd ../prime
rsync -auv --include="*/" --include="*onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-mutation_score-ratio_rank*" --exclude="*" --exclude="*onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-mutation_score-ratio_rank-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_prime_fixed/ "./best_primeac/" --progress
cd ./best_primeac
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-mutation_score-ratio_rank-*
rmdir bootstrapping
rmdir raw
touch onehot_Mask_Peptide_EL_rank_mut_dissimilarity_score-mutation_score-ratio_rank