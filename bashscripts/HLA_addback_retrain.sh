#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221229_HLA_addback_retrain/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./HLA_addback_retrain.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -ncores 39

