#!/bin/bash

fairseq-train data-bin \
    --restore-file ./checkpoints/checkpoint_last.pt \
    --arch transformer \
    --reset-optimizer \
    --max-epoch 20 \
    --fp16 \
    --max-tokens 5000 \
    --optimizer adam \
    --lr 1e-4 \
    --dropout 0.2 \
    --no-epoch-checkpoints