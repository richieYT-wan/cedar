#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR= "${DIR}test_data/"


# Kmers extraction script
cd $DIR/scripts
python generate_kmers.py -filepath $DATADIR -k $1 -outdir $OUTDIR -description_verbose "false" -drop_sequence "true"
