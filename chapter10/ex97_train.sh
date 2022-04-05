#!/bin/bash
if [ -d ./checkpoints ]; then
    rm -r checkpoints
fi

fairseq-train data-bin \
    --arch transformer \
    --reset-optimizer \
    --max-epoch 30 \
    --max-tokens 5000 \
    --optimizer adam \
    --lr 1e-4 \
    --dropout 0.2 \
    --fp16 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --no-epoch-checkpoints