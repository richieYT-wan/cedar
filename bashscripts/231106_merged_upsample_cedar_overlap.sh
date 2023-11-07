#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
INDEX=$2
INPUTTYPE=icore_mut
TRAINSET=merged_upsampled_overlap
WALLTIME="00:45:00"

# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/231107_upsample_prime_wC_overlap/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"$TRAINSET\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir \${ICSDIR} -ncores 2 -outdir \${OUTDIR} -trainset ${TRAINSET} -condition ${CONDITION} -input_type ${INPUTTYPE} -mc_index ${INDEX}
" > "231018_redo_cdt_${CONDITION}_trainset_${TRAINSET}_index_${INDEX}_input-type_${INPUTTYPE}.sh"

chmod +x "231018_redo_cdt_${CONDITION}_trainset_${TRAINSET}_index_${INDEX}_input-type_${INPUTTYPE}.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=3:thinnode,mem=12gb,walltime=$WALLTIME "./231018_redo_cdt_${CONDITION}_trainset_${TRAINSET}_index_${INDEX}_input-type_${INPUTTYPE}.sh"

rm "./231018_redo_cdt_${CONDITION}_trainset_${TRAINSET}_index_${INDEX}_input-type_${INPUTTYPE}.sh"