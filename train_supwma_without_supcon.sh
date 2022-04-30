#!/bin/bash

## SupWMA without contrastive learning
input_data=198ol_602other  # 198gt-swm + 198ol-swm + 602other (198 swm clusters, 198 swm outliers, 602 others)
input_path=./TrainData/198gtswm_${input_data}/h5_np15
out_path=./ModelWeights/supwma_without_supcon
epoch=100

# only eval on fold zero
#python train_k_fold.py --eval_fold_zero --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --opt Adam --train_batch_size 1024 --val_batch_size 1024 --lr 1e-3 --weight_decay 0 --scheduler step --step_size 20 --redistribute_class

# 5-fold cross validation
python train_k_fold.py --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --opt Adam --train_batch_size 1024 --val_batch_size 1024 --lr 1e-3 --weight_decay 0 --scheduler step --step_size 20 --redistribute_class
