#! /usr/bin/bash
# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

cd ../pyscripts/
for k in 12
do
	for chunk in 0 1 2 3
	do
		python score_kmers_hla.py -filepath "../output/${k}mers_humanproteome_chunk_${chunk}.txt" -resultspath "../output_xls/" -threshold 20.0
	done
done
