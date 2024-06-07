#!/bin/bash

export TRANSFORMERS_VERBOSITY=info
export DS_SKIP_CUDA_CHECK=0

torchrun --nproc_per_node=4 --master_port=2345 ../src/src/train.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --data_path ../data/mvd_rebuilt_50k_x86_4_tasks_split \
    --output_dir ../x86_codellama \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --fp16 \
    --gradient_checkpointing \
    --deepspeed "../configs/ds_config.json" \
    --report_to "wandb" \
    --logging_steps 5 \
    --eval_steps 10 \
    --do_eval True \
    --run_name "x86_codellama_v1" 

