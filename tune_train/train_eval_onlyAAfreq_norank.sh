#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/230112_AAfreq_trueNORANK/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./train_eval_norank.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -ncores 39  > 'onlyAAfreq_norank.txt'

