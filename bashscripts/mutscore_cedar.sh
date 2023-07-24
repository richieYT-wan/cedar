#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221122_mutscore_cedar_fixed/
ICSDIR=../data/ic_dicts/
TRAINSET="cedar"
cd ${PYDIR}
pwd
python3 ./mutscores_train_eval.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -trainset ${TRAINSET} -ncores 39  > 'tmp.txt'

