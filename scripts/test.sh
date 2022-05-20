#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}/output/"
DATADIR= "${DIR}test_data/"


# Kmers extraction script
python $DIR/scripts/generate_kmers.py -filepath $DATADIR/$1/ -k $k -description_verbose "false" -drop_sequence "true"

# concatenate all called k-mers into a merged_file
cat *${k}mers*.txt > "${k}mers_$1_merged.txt"

# Remove all the other split files
rm *${k}mers_Human_split*.txt

# will keep only peptide and export to pep format
awk -F ',' 'NR>1 {print $1}' ${k}mers_human_merged.txt  >> ${k}mers_human.pep

