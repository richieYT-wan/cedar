#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
INDEX=$2
INPUTTYPE=$3
WALLTIME="00:25:00"
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/230427_MutExpr_Final_input_type/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"cedar\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./2305XX_final_mutExpr_single_mc_newratio.py -icsdir \${ICSDIR} -trainset sine -ncores 1 -outdir \${OUTDIR} -condition None -key EL_rank_mut -input_type icore
" > "230427_MCFinal_${1}_exp8mers_index_${2}_input_type_${3}.sh"

chmod +x "230427_MCFinal_${1}_exp8mers_index_${2}_input_type_${3}.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=1:thinnode,mem=10gb,walltime=$WALLTIME "./230427_MCFinal_${1}_exp8mers_index_${2}_input_type_${3}.sh"
rm "230427_MCFinal_${1}_exp8mers_index_${2}_input_type_${3}.sh"
