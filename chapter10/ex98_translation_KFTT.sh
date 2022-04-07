fairseq-generate data-bin \
    --batch-size 64 \
    --path ./checkpoints/checkpoint_best.pt \
    < ./datasets/kftt-data-1.0/data/spm_tok/kyoto-test-spm.ja \
    | grep '^H-' \
    | sort -V \
    | cut -f 3 \
    > ./outputs/ex98_outputs/ex98_output_KFTT_spm.en