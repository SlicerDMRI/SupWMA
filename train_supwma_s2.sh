#!/bin/bash

# SupWMA
opt='Adam'
wd=0
momentum=-1
contra_lr=1e-2
contra_epoch=100
cls_lr=1e-3
cls_epoch=80
supcon_epoch=100
save_step=50

scheduler='wucd'
T_0=110
T_mult=100
cls_T_0=10
cls_T_mult=2

train_batch_size=6144
cls_train_batch_size=1024
tmp=0.1

num_swm_ol=198  # 198gt-swm + 198ol-swm
input_path=./TrainData_TwoStage/stage2/${num_swm_ol}outlier/h5_np15/  # stage 2
entire_data_path=./TrainData_TwoStage/entire_data_two_stage/h5_np15/  # for evaluation (1 million streamlines)

s1_path=./ModelWeights/supwma_stage1_weights
encoder_path=./ModelWeights/supwma_stage2_encoder_weights
classifier_path=/supwma_stage2_cls_weights

# only eval on fold zero
# python train_s2_contrastive.py --eval_fold_zero --input_path ${input_path} --epoch ${contra_epoch} --out_path_base ${encoder_path} --save_step ${save_step} --opt ${opt} --momentum ${momentum} --lr ${contra_lr} --weight_decay ${wd} --train_batch_size ${train_batch_size} --scheduler ${scheduler}  --T_0 ${T_0} --T_mult ${T_mult} --head_name mlp --encoder_feat_num 128 --temperature ${tmp}
python train_s2_classifier.py --input_path ${input_path} --epoch ${cls_epoch} --out_path_base ${encoder_path}${classifier_path} --supcon_epoch ${supcon_epoch} --opt ${opt} --train_batch_size ${cls_train_batch_size} --val_batch_size 4096 --lr ${cls_lr} --weight_decay ${wd} --scheduler ${scheduler} --T_0 ${cls_T_0} --T_mult ${cls_T_mult} --redistribute_class --stage1_weight_path_base ${s1_path} --input_eval_data_path ${entire_data_path}

# # Five fold cross-validation
# python train_s2_contrastive.py --input_path ${input_path} --epoch ${contra_epoch} --out_path_base ${encoder_path} --save_step ${save_step} --opt ${opt} --momentum ${momentum} --lr ${contra_lr} --weight_decay ${wd} --train_batch_size ${train_batch_size} --scheduler ${scheduler}  --T_0 ${T_0} --T_mult ${T_mult} --head_name mlp --encoder_feat_num 128 --temperature ${tmp}
# python train_s2_classifier.py --input_path ${input_path} --epoch ${cls_epoch} --out_path_base ${encoder_path}${classifier_path} --supcon_epoch ${supcon_epoch} --opt ${opt} --train_batch_size ${cls_train_batch_size} --val_batch_size 4096 --lr ${cls_lr} --weight_decay ${wd} --scheduler ${scheduler} --T_0 ${cls_T_0} --T_mult ${cls_T_mult} --redistribute_class --stage1_weight_path_base ${s1_path} --input_eval_data_path ${entire_data_path}



