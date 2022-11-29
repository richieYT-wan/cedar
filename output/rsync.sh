#! /usr/bin/bash

DIR="/Users/riwa/Documents/code/cedar/output/"
for file in 221122_mutscore_merged_fixed 221122_mutscore_prime_fixed
do
	echo $file
	$(mkdir "${DIR}${file}/")
	$(mkdir "${DIR}${file}/bootstrapping/")
	rsync -aur --progress "yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/${file}/total_df.csv" "${DIR}${file}/${file}_total_df.csv"
	#rsync -aur --progress "yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/${file}/bootstrapping/*mean_rocs*" "${DIR}${file}/bootstrapping/"
done
