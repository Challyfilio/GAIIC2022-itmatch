#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export TOKENIZERS_PARALLELISM=false
echo '* * * * * * *'

python -u src/training/main.py \
  --save-frequency 1 \
  --train-data="src/data/train_coarse.txt,src/data/train_fine.txt.00" \
  --val-data="src/data/train_fine.txt.01" \
  --dataset-type="json" \
  --model="ViT-B/32" \
  --warmup 1000 \
  --batch-size=128 \
  --lr=1e-4 \
  --wd=0.1 \
  --epochs=10 \
  --workers=4 \
  --dp \
  --multigpu 0 \
  --name "demo"
