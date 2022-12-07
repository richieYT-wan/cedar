#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221122_mutscore_merged_fixed/
ICSDIR=../data/ic_dicts/
CDTDIR=../output/best_conditions/
TRAINSET="merged"
cd ${PYDIR}
pwd
python3 ./mutscores_train_eval_norank.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -cdtdir ${CDTDIR} -ncores 39  > 'tmp.txt'

