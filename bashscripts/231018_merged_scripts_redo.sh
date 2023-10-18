#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
INDEX=$2
INPUTTYPE=$3
trainset=$4
WALLTIME="00:30:00"
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/231018_redo_merged/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"cedar\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
trainset=\"$trainset\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir \${ICSDIR} -trainset \${trainset} -ncores 1 -outdir \${OUTDIR} -condition
\${CONDITION} -key None -mc_index \${INDEX} -input_type \${INPUTTYPE}
" > "231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"

chmod +x "231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=10:thinnode,mem=10gb,walltime=$WALLTIME "./231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"
rm "231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"
