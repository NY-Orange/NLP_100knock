#!/bin/bash
if [ -d ./checkpoints ]; then
    rm -r checkpoints
fi

fairseq-train data-bin \
    --arch transformer \
    --reset-optimizer \
    --max-epoch 3 \
    --fp16 \
    --max-tokens 5000 \
    --optimizer adam \
    --lr 1e-4 \
    --dropout 0.2 \
    --no-epoch-checkpoints