#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/230724_TrainDataPruning_CompareModels/
ICSDIR=../data/ic_dicts/
TRAINSET="cedar"
cd ${PYDIR}
python3 ./traindata_pruning.py -icsdir ${ICSDIR} -trainset ${TRAINSET} -ncores 39 -outdir ${OUTDIR}

