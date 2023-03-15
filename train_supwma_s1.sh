#!/bin/bash

## Stage 1 training 
s1_epoch=80
s1_opt='Adam'
s1_lr=1e-2
s1_wd=0
s1_path="./ModelWeights/supwma_stage1_weights"

# 198 SWM + 602 Other
input_data=198swm_602other
input_path=./TrainData_TwoStage/stage1/${input_data}/h5_np15

# only eval on fold zero
python train_s1.py --eval_fold_zero --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --opt ${s1_opt} --train_batch_size 1024 --val_batch_size 4096 --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2

# # 5-fold cross validation
# python train_s1.py --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --opt ${s1_opt} --train_batch_size 1024 --val_batch_size 4096 --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2