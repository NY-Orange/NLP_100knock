#!/bin/bash
if [ -d ./data-bin ]; then
    rm -r data-bin/
fi

fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref datasets/en-ja/spm_tok/JPara-train-5000000-spm \
    --validpref datasets/kftt-data-1.0/data/spm_tok/kyoto-dev-spm \
    --testpref datasets/kftt-data-1.0/data/spm_tok/kyoto-test-spm \
    --joined-dictionary \
    --nwordssrc 30000 \
    --nwordstgt 30000