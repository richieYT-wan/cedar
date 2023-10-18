#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="KL-Mask"
INDEX=1023
INPUTTYPE="icore_mut"
TRAINSET="cp_merged"
WALLTIME="01:00:00"
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/231018_redo_merged/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"$TRAINSET\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 1 -outdir \${OUTDIR} -condition \${CONDITION} -key None -input_type \${INPUTTYPE} -mc_index \${INDEX}
" > "231018_timebenchmark_1core_cedar-prime.sh"

chmod +x "231018_timebenchmark_1core_cedar-prime.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=1:thinnode,mem=10gb,walltime=$WALLTIME "./231018_timebenchmark_1core_cedar-prime.sh"


# Set the variable CONDITION to the first positional argument
CONDITION="KL-Mask"
INDEX=1023
INPUTTYPE="icore_mut"
TRAINSET="cpn_merged"
WALLTIME="01:00:00"
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/231018_redo_merged/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"$TRAINSET\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
INPUTTYPE=\"$INPUTTYPE\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./231018_mutexpr_merged_trainsets.py -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 1 -outdir \${OUTDIR} -condition \${CONDITION} -key None -input_type \${INPUTTYPE} -mc_index \${INDEX}
" > "231018_timebenchmark_1core_cedar-prime-nepdb.sh"

chmod +x "231018_timebenchmark_1core_cedar-prime-nepdb.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=1:thinnode,mem=10gb,walltime=$WALLTIME "./231018_timebenchmark_1core_cedar-prime-nepdb.sh"