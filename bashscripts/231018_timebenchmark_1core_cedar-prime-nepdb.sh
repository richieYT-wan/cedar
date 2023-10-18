#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/231018_redo_merged/
ICSDIR=${HOMEDIR}/data/ic_dicts/
TRAINSET="cpn_merged"
CONDITION="KL-Mask"
INDEX="1023"
INPUTTYPE="icore_mut"
cd ${PYDIR}
pwd
echo "Starting PyScript"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir ${ICSDIR} -ncores 1 -outdir ${OUTDIR} -trainset cpn_merged -condition KL-Mask -input_type icore_mut -mc_index 1023

