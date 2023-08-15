#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/230207_aligned_exprscore_addwtrank/
ICSDIR=../data/ic_dicts/
TRAINSET="cedar"
cd ${PYDIR}
pwd
python3 ./aligned_icore_exprscore_mutscore_train_eval.py -icsdir ${ICSDIR} -trainset ${TRAINSET} -add_wtrank True -ncores 39  > 'tmp.txt'

