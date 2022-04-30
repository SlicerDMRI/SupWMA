#!/bin/bash

## SupWMA
opt='Adam'
wd=0
momentum=-1
contra_lr=1e-2
contra_epoch=100
cls_epoch=80
supcon_epoch=100
save_step=50

scheduler='wucd'
T_0=110
T_mult=1

train_batch_size=6144
tmp=0.1

# 198gt-swm + 198ol-swm + 602other
num_swm_ol=198
num_other=602

input_path=./TrainData/198gtswm_${num_swm_ol}ol_${num_other}other/h5_np15/
encoder_path=./ModelWeights/supwma_with_supcon/contra_${num_swm_ol}ol_${num_other}other_epoch${contra_epoch}_${opt}_lr${contra_lr}_wd${wd}_mon${momentum}_${scheduler}${T_0}_${T_mult}_tmp${tmp}
classifier_path=/encoder${supcon_epoch}epoch_classifier_wucd_baseline_${cls_epoch}epoch

# only eval on fold zero
python train_contrastive_encoder.py --eval_fold_zero --input_path ${input_path} --epoch ${contra_epoch} --out_path_base ${encoder_path} --save_step ${save_step} --opt ${opt} --momentum ${momentum} --lr ${contra_lr} --weight_decay ${wd} --train_batch_size ${train_batch_size} --scheduler ${scheduler}  --T_0 ${T_0} --T_mult ${T_mult} --head_name mlp --encoder_feat_num 128 --temperature ${tmp}
python train_classifier.py --input_path ${input_path} --epoch ${cls_epoch} --out_path_base ${encoder_path}${classifier_path} --supcon_epoch ${supcon_epoch} --opt Adam --train_batch_size 1024 --val_batch_size 4096 --lr 1e-3 --weight_decay 0 --scheduler wucd --T_0 10 --T_mult 2 --redistribute_class


# Five fold cross-validation
python train_contrastive_encoder.py --input_path ${input_path} --epoch ${contra_epoch} --out_path_base ${encoder_path} --save_step ${save_step} --opt ${opt} --momentum ${momentum} --lr ${contra_lr} --weight_decay ${wd} --train_batch_size ${train_batch_size} --scheduler ${scheduler}  --T_0 ${T_0} --T_mult ${T_mult} --head_name mlp --encoder_feat_num 128 --temperature ${tmp}
python train_classifier.py --input_path ${input_path} --epoch ${cls_epoch} --out_path_base ${encoder_path}${classifier_path} --supcon_epoch ${supcon_epoch} --opt Adam --train_batch_size 1024 --val_batch_size 4096 --lr 1e-3 --weight_decay 0 --scheduler wucd --T_0 10 --T_mult 2 --redistribute_class