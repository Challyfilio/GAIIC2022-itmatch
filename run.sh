#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export TOKENIZERS_PARALLELISM=false

python -u src/training/main.py \
    --save-frequency 3   \
    --train-data="src/data/train_coarse.txt,src/data/train_fine.txt.00"    \
    --val-data="src/data/train_fine.txt.01"   \
    --dataset-type="json" \
    --warmup 1000  \
    --batch-size=256  \
    --lr=1e-4  \
    --wd=0.1  \
    --epochs=15  \
    --workers=4   \
    --dp   \
    --multigpu 0 \
    --name "demo3"