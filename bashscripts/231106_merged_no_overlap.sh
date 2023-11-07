#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
INDEX=$2
INPUTTYPE=$3
TRAINSET=$4
if [ "$TRAINSET" == "cp_merged" ]; then
    WALLTIME="00:12:00"
elif [ "$TRAINSET" == "cpn_merged" ]; then
    WALLTIME="00:10:00"
else
    # Default value if TRAINSET doesn't match either option
    WALLTIME="00:09:00"
fi
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/231106_reredo_merged_no_overlap/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"$TRAINSET\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir \${ICSDIR} -ncores 2 -outdir \${OUTDIR} -trainset merged_no_overlap -condition ${CONDITION} -input_type ${INPUTTYPE} -mc_index ${INDEX}
" > "231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"

chmod +x "231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=2:thinnode,mem=8888mb,walltime=$WALLTIME "./231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"

rm "./231018_redo_cdt_${1}_trainset_${4}_index_${2}_input-type_${3}.sh"