#!/bin/csh

set WDIR = /home/projects/vaccine/people/morni/netMHCIIpan_retrain/4.1_DRupdate_DQ4corrected/train_BergsengSA_wo_context

set NNLIN = "/home/projects/vaccine/people/morni/bin/nnalign_gaps_pan_add_ligands_two_outputs_context_allelelist_v10.5-1"
set PSEUDO = "data/pseudosequence.2016.all.X.dat"
set ALLELELIST = "data/allelelist"
set RANPFILE = "/home/projects/vaccine/people/birey/nar/dat/pipeline_2020_04_01/NNalignIn/rand15mers_1K.txt"
set MPAT = "/home/projects/vaccine/people/morni/matrices/BLOSUM50"
set BLF = "/home/projects/vaccine/people/morni/matrices/blosum62.freq_rownorm"

mkdir -p $WDIR/SYN
mkdir -p $WDIR/TSYN

# Iterates through partitions
foreach n ( 0 1 2 3 4 )

	set ba_testdata=data/c00$n"_ba"
	set el_testdata=data/c00$n"_el"
	set ba_traindata=data/f00$n"_ba"
	set el_traindata=data/f00$n"_el"

# Iterates through hidden nueron ensambles
	foreach nhid ( 20 40 60 )
	# Iterates through seeds
		foreach s ( 1 2 3 4 5 6 7 8 9 10 )

			set SYN=SYN/$n.$nhid.$s.syn.bl
			set TSYN=TSYN/$n.$nhid.$s.syn.bl

			if ( ! -e $SYN ) then 

				touch $SYN

				echo '#\!/bin/csh' > $n.$nhid.$s.csh
				echo "$NNLIN -rpepfile $RANPFILE -aX -bl -p1a -burn_p1a 2 -burnin_HLA 20 -elpfr 0 -eplen -1 -fl 3 -i 400 -l 9 -nh 20 -ns 300000 -nt 1 -s $s -alternate 0.50000 -allelelist $ALLELELIST -blf $BLF -classI 13,14,15,16,17,18,19,20,21 -syn $SYN -nh $nhid -mhc $PSEUDO -ft $ba_testdata -test_EL $el_testdata -train_EL $el_traindata $ba_traindata -tsyn $TSYN -mpat $MPAT > out.$n.$nhid.$s" >> $n.$nhid.$s.csh
				echo "# EOF" >> $n.$nhid.$s.csh

				qsub -d $WDIR -W group_list=vaccine -A vaccine -l nodes=1:ppn=1,mem=26gb,walltime=06:00:00:00 $n.$nhid.$s.csh


				#break 1

				sleep 0.1

			endif
		end
	end
end
