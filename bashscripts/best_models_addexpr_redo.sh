#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/230410_redo_AddExpression_BestModels/
ICSDIR=../data/ic_dicts/
TRAINSET="cedar"
cd ${PYDIR}
pwd
python3 ./best_models_addexpr.py -icsdir ${ICSDIR} -trainset ${TRAINSET} -ncores 39 -outdir ${OUTDIR}

