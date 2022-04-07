fairseq-generate data-bin \
    --batch-size 64 \
    --path ./checkpoints/checkpoint_best.pt \
    < ./datasets/kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H-' \
    | sort -V \
    | cut -f 3 \
    > ./outputs/ex92_outputs/ex92_output.en