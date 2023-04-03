#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
OUTDIR=${HOMEDIR}output/viral_featimp_sampling/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./aligned_icore_exprscore_mutscore_train_eval.py -icsdir ${ICSDIR} -ncores 39  > 'tmp.txt'

