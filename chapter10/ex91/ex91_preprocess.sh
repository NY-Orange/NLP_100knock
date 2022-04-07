#!/bin/bash
if [ -d ./data-bin ]; then
    rm -r data-bin/
fi

fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref datasets/kftt-data-1.0/data/tok/kyoto-train.cln \
    --validpref datasets/kftt-data-1.0/data/tok/kyoto-dev \
    --testpref datasets/kftt-data-1.0/data/tok/kyoto-test \
    --joined-dictionary \
    --nwordssrc 30000 \
    --nwordstgt 30000