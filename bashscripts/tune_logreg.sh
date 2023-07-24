#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/partitioned_traindata/
OUTDIR=../output/tuning/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./tune_aafreq_logreg.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -debug False>'xd_logreg.txt'

