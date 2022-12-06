#! /usr/bin/bash

for mask in 'false' 'C'
do
	for trainset in 'cedar' 'prime' 'merged'
	do
		filename="/home/projects/vaccine/people/yatwan/cedar/tune_train/mutscores_nn_train${trainset}mask${mask}"
		touch ${filename}
		echo "#! /usr/bin/bash" > ${filename}
		echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> ${filename}                                            
		echo "source activate phd" >> ${filename}
		echo "HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/" >> ${filename}
		echo "PYDIR=\${HOMEDIR}pyscripts/" >> ${filename}
		echo "DATADIR=../data/mutant/" >> ${filename}
		echo "OUTDIR=../output/221206_mutscores_NN_${trainset}_mask${mask}/" >> ${filename}
		echo "ICSDIR=../data/ic_dicts/" >> ${filename}
		echo "TRAINSET="${trainset}"" >> ${filename}
		echo "cd \${PYDIR}" >> ${filename}
		echo "pwd" >> ${filename}
		echo "python3 ./mutscores_train_eval.py -datadir \${DATADIR} -outdir \${OUTDIR} -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 39  > 'tmp.txt'" >> ${filename}
	done
done