#!/bin/bash

for BEAM_WIDTH in 1 10 20 30 40 50 60 70 80 90; do
    fairseq-generate data-bin \
        --batch-size 16 \
        --beam $BEAM_WIDTH \
        --path ./checkpoints/checkpoint_best.pt \
        < ./datasets/kftt-data-1.0/data/spm_tok/kyoto-test-spm.ja \
        | grep '^H-' \
        | sort -V \
        | cut -f 3 \
        > ./outputs/ex95_outputs/ex95_output_beam${BEAM_WIDTH}_spm.en
done