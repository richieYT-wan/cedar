#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd
HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/partitioned_traindata/
ICSDIR=../data/ic_dicts/ 
OUTDIR=../output/train_eval/
cd $PYDIR
pwd
python3 ./nested_kcv_aafreq_methods.py -ncores 36 -model log -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -tunedir ${HOMEDIR}output/tune_log/ -debug False>'traineval_logreg.txt'
