#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221122_new_core_mutscores_trainCedar_maskcysteine/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 -W ignore ./new_mutscores_cedar_maskcysteine.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -ncores 38

