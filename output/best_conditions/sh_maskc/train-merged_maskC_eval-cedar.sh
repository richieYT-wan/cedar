#! /usr/bin/bash
mkdir ../merged_maskC
cd ../merged_maskC
rsync -auv --include="*/" --include="*onehot_Inverted-Shannon_icore_mut_EL_rank_mut_aa_props*" --exclude="*" --exclude="*onehot_Inverted-Shannon_icore_mut_EL_rank_mut_aa_props-" yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/221122_mutscore_merged_maskcysteine/ "./best_cedar/" --progress
cd ./best_cedar
rm ./raw/*preds*
rm ./bootstrapping/*bootstrapped_df*
mv ./raw/* ./
mv ./bootstrapping/* ./
rm *onehot_Inverted-Shannon_icore_mut_EL_rank_mut_aa_props-*
rmdir bootstrapping
rmdir raw
touch onehot_Inverted-Shannon_icore_mut_EL_rank_mut_aa_props