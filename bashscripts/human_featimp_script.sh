#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
OUTDIR=${HOMEDIR}output/230426_human_featimp_sampling_interative/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./adding_humanprot.py -icsdir ${ICSDIR} -outdir ${OUTDIR} -ncores 40

