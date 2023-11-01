#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
OUTDIR=${HOMEDIR}output/231101_redo_viral_merged_dataset/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./231101_redo_adding_virus_merged.py -icsdir ${ICSDIR} -outdir ${OUTDIR} -ncores 40  > 'virout.txt'

