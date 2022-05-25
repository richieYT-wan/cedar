#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

# $1 should be the input directory or filepath, relative to DIR
# ex, if dir is ./cedar/data/human_split/, then $1 should be "data/human_split/"
# $2 should be the target filename: i.e. "human" if running for human
#
# TODO: $4 should be the split number when implementing with splitting from a total file

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}data/human_split/"
SCRIPTDIR="${DIR}kmers_selfsimilarity/"
NAME="humanproteome"
for k in 8 9 10 11 12 15
do
  cd $SCRIPTDIR
  # This will save the output files as OUTDIR/kmers_NAME_chunk_X
  sh kmer_chunk.sh ${k} 4 ${NAME}
  for chunk in 0 1 2 3
  do
    sh script_netmhcpan_chunk.sh ${k} ${chunk} ${NAME} "${OUTDIR}${k}mers_${NAME}_chunk_${chunk}.pep"
  done
done


