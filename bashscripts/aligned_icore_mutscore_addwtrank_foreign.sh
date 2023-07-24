#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
OUTDIR=${HOMEDIR}output/230207_aligned_mutscore_addwtrank_foreignness/
TRAINSET="cedar"
cd ${PYDIR}
pwd
python3 ./aligned_icore_mutscore_train_eval.py -icsdir ${ICSDIR} -trainset ${TRAINSET} -ncores 39 -add_wtrank True -add_foreignness True -outdir ${OUTDIR}> 'tmp.txt'

