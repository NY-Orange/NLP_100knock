fairseq-generate data-bin \
    --batch-size 16 \
    --beam 50 \
    --path ./checkpoints/checkpoint_best.pt \
    < ./datasets/kftt-data-1.0/data/spm_tok/kyoto-test-spm.ja \
    | grep '^H-' \
    | sort -V \
    | cut -f 3 \
    > ./outputs/ex97_outputs/ex97_output_spm.en