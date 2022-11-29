#! /usr/bin/bash

for train in "cedar" "prime
do
	for supp in "" "maskcysteine"
		do
			echo "redosave_${train}_${supp}"
		done
done

