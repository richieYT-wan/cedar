#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}test_data/"
SCRIPTDIR="${DIR}scripts/"

# Kmers extraction script
for k in 8 9 10
do
  cd SCRIPTDIR
  python generate_kmers.py -filepath $DATADIR -k $k -outdir $OUTDIR -description_verbose "false" -drop_sequence "true"
  cd OUTDIR
  # concatenate all kmers split
  cat *${k}mers*.txt > "${k}mers_merged.txt"
  # Remove all the other split files
  rm $OUTDIR*$split*.txt
  awk -F ',' 'NR>1 {print $1}' ${k}mers_merged.txt  >> ${k}mers_${1}.pep
done



