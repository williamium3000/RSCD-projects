#!/usr/bin/env bash
set -x
PARTITION=$1
#GPUs
gpus=0

#Set paths
checkpoint_root=./checkpoints
vis_root=./vis
data_name=S2Looking


img_size=256    
batch_size=8   
lr=0.01         
max_epochs=200

net_G=base_transformer_pos_s4_dd8    

lr_policy=linear
optimizer=sgd                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
# pretrain=pretrain/pretrained_changeformer.pt

#Train and Validation splits
split=train         #trainval
split_val=val      #test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

# CUDA_VISIBLE_DEVICES=1 
srun -p ${PARTITION} \
        --job-name changeformer_S2Looking256 \
        --gres=gpu:1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --kill-on-bad-exit=1 python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr}
