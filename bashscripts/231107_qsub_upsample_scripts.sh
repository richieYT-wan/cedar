#!/bin/bash

cd /home/projects/vaccine/people/yatwan/cedar/bashscripts/
for cdt in KL-Mask KL None Inverted-KL-Mask Inverted-KL
do
  for index in $(seq 0 1)
  do
    sh 231106_merged_upsample_cedar_overlap.sh ${cdt} ${index}
    sh 231106_merged_upsample_no_overlap.sh ${cdt} ${index}
  done
done

