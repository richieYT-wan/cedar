#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/230109_all_a11a03_shuffle_icore_mut/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./mutscores_train_eval_all_a11a03_posnegshuffle.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -input_type icore_mut -ncores 39  > 'cedar_noa11a03_icoremut.txt'

