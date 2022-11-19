#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221119_new_core_mutscores_primeTrainset/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./new_mutscores_prime.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -ncores 38 > 'tmp.txt'

