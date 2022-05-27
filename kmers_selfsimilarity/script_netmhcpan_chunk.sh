#! /usr/bin/bash 

# $1 should be K of kmers
# $2 should be N of n_chunks
# $3 should be name given (ex "humanproteome")
# $4 should be path to .pep file

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}data/human_split/"
SCRIPTDIR="${DIR}kmers_selfsimilarity/"
TMPDIR="${SCRIPTDIR}tmp/"


if [[ ! -d "${TMPDIR}" ]]
then
  mkdir ${TMPDIR}
fi

for k in 8 9 10 11 12 15
do
  for chunk in 0 1 2 3
  do
    for hla in 'HLA-A02:01' 'HLA-B07:02' 'HLA-A03:01' 'HLA-A24:02' 'HLA-A11:01' 'HLA-B15:01' 'HLA-B35:01' 'HLA-A02:06' 'HLA-B27:05' 'HLA-A01:01'
    do
    	hlaname="${hla//:}"
    	filename="${hlaname}_${1}mers_${3}_chunk_${2}"
    	pepname="${OUTDIR}${k}mers_humanproteome_chunk_${chunk}.pep"
    	touch "${TMPDIR}${filename}.sh"
    	echo "#\! /usr/bin/bash" >> "${TMPDIR}${filename}.sh"
    	echo PBS -N ${filename}
    	echo netMHCpan-4.1 -a ${hla} -p ${pepname} -t 2.0 -BA -xls -xlsfile "${OUTDIR}${filename}.xls" > "${TMPDIR}${filename}.sh"
    	echo "# EOF" >> "${TMPDIR}${filename}.sh"
    	qsub -d "${DIR}" -W group_list=vaccine -A vaccine -l nodes=1:ppn=1:thinnode,mem=12gb,walltime=3:00:00 "${TMPDIR}${filename}.sh"
    done
  done
done


