#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

DIR = "/home/projects/vaccine/people/yatwan/kmers_pipeline/"
OUTDIR = "/home/projects/vaccine/people/yatwan/kmers_pipeline/output/"
DATADIR = "/home/projects/vaccine/people/yatwan/data/"

# awk 'BEGIN {n_seq=0;} /^>/ {if(n_seq%1000==0){file=sprintf("tmp_split_%d.fa",n_seq);} print >> file; n_seq++; next;} { print >> file; }' < $DATADIR$1

# PUT FASTA SPLIT SCRIPT HERE
python $DIR/scripts/split_fasta.py -filepath $DATADIR/$1

# $1 should be input folder which will also give the name
for k in 8 9 10 11
do
  # Kmers extraction script
  python $DIR/scripts/generate_kmers.py -filepath $DATADIR/$1/ -k $k -description_verbose "false" -drop_sequence "true"
  # concatenate all called k-mers into a merged_file
  # name=*${k}mers*.txt
  cat *${k}mers*.txt > "${k}mers_$1_merged.txt"
  # Remove all the other split files
  rm *${k}mers_Human_split*.txt
  # will keep only peptide and export to pep format
  awk -F ',' 'NR>1 {print $1}' ${k}mers_human_merged.txt  > ${k}mers_human.pep

  netMHCpan-4.1 -a HLA-A01:01 HLA-A02:01 HLA-A03:01 HLA-A24:01 HLA-B07:02 HLA-B08:01 HLA-B24:01 -BA 1 -p