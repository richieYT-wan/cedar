#! /usr/bin/bash
mkdir ../merged
cd ../merged
rsync -auv --include="*/" --include="*onehot_Inverted-KL_Peptide_EL_rank_mut_aa_props*" --exclude="*" --exclude="*onehot_Inverted-KL_Peptide_EL_rank_mut_aa_props-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_merged_fixed/ "./best_cedar/" --progress
cd ./best_cedar
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Inverted-KL_Peptide_EL_rank_mut_aa_props-*
rmdir bootstrapping
rmdir raw
touch onehot_Inverted-KL_Peptide_EL_rank_mut_aa_props