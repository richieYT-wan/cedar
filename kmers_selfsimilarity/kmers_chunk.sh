#! /usr/bin/bash
## Activates conda environment
#source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
#source activate phd

# generates kmers for a length K given by $1
# Splits the results into ${2} chunks
# Saves the files with the identifier ${1}mers_${3}_chunk_X.txt
DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}data/human_split/"
PYDIR="${DIR}pyscripts/"

# Do CD here because somehow the relative imports don't work properly in the .py script
# when running the script as python ./scripts/generate_kmers.py
cd ${PYDIR}
python generate_kmers.py -filepath $DATADIR -k ${1} -outdir $OUTDIR -description_verbose "false" -drop_sequence "true"
cd ${OUTDIR}
# concatenate the splits by finding matching ${1}mers in .txt and "split"
cat $(find . -name "*${1}mers*.txt" -and -name "*split*") > "${1}mers_${3}.txt"
# Remove the splits after concatenation
rm $(find . -name "*${1}mers*.txt" -and -name "*split*")
# Dedupe here + saving merged to .txt and .pep, split output into ${2} chunks
python "${PYDIR}remove_duplicates.py" -filepath "${OUTDIR}${1}mers_${3}.txt" -split "True" -n ${2}

# output files will be named ${OUTDIR}${1}mers_${3}_chunk_X.pep and .txt