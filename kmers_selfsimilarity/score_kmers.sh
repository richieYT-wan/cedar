#! /usr/bin/bash
# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR="/home/projects/vaccine/people/yatwan/cedar/"
PYDIR="${HOMEDIR}pyscripts/"
DATADIR="${HOMEDIR}output/"

for ch in 0 1 2 3
do
  for k in 8 9 10 11 12
	do
	  echo "CHUNK: ${ch}"
	  python score_kmers_hla.py -filepath "${DATADIR}12mers_humanproteome_chunk_${ch}.txt" -resultspath "${DATADIR}12mers/" -threshold 20.0 &
	  done
done
