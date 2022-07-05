#! /usr/bin/bash
#source /home/riwa/unixdir/anaconda3/etc/profile.d/conda.sh


DIR='/home/riwa/unixdir/code/cedar/output_xls/'
PYSCRIPT=~/unixdir/code/cedar/pyscripts/score_kmers_hla.py
FILEPATH=~/unixdir/code/cedar/output/*chunk*.txt
for x in $(ls ${FILEPATH} | grep -v scored);
do
	echo ${x} 
	echo "HERE IS DIR" ${DIR}
	python ${PYSCRIPT} -filepath ${x} -resultspath "${DIR}" -outdir "${DIR}"
done
