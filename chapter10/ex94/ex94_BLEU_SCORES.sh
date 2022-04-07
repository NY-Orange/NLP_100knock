#!/bin/bash

for BEAM_WIDTH in 1 10 20 30 40 50 60 70 80 90 100; do
    fairseq-score \
        --sys ./outputs/ex94_outputs/ex94_output_beam$BEAM_WIDTH.en \
        --ref ./datasets/kftt-data-1.0/data/tok/kyoto-test.en \
    > ./outputs/ex94_outputs/ex94_output_BLUE_SCORE_beam$BEAM_WIDTH.txt
done