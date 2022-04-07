#!/bin/bash
if [ -d ./checkpoints ]; then
    rm -r checkpoints
fi

fairseq-train data-bin \
    --arch transformer \
    --reset-optimizer \
    --max-epoch 5 \
    --max-tokens 5000 \
    --optimizer adam \
    --lr 1e-4 \
    --dropout 0.2 \
    --fp16 \
    --no-epoch-checkpoints \
    --tensorboard-logdir ./outputs/ex96_outputs/ex96_log \
    > ./outputs/ex96_outputs/ex96.log