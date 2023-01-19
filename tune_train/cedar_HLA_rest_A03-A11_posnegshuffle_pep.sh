#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/230119_rest_AXX_posnegshuffle/
ICSDIR=../data/ic_dicts/
INPUT_TYPE=Peptide
cd ${PYDIR}
pwd
python3 ./HLA_rest_A03-A11_posnegshuffle.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -input_type ${INPUT_TYPE} -ncores 39

