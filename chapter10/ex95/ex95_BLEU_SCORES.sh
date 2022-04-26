#!/bin/bash

for BEAM_WIDTH in 1 10 20 30 40 50 60 70 80 90; do
    fairseq-score \
        --sys ./outputs/ex95_outputs/ex95_output_beam$BEAM_WIDTH.en \
        --ref ./datasets/kftt-data-1.0/data/orig/kyoto-test.en \
    > ./outputs/ex95_outputs/ex95_output_BLUE_SCORE_beam$BEAM_WIDTH.txt
done