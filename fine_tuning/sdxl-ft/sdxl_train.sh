export CUDA_VISIBLE_DEVICES=$1

accelerate launch --main_process_port=$3 --num_processes=$2 --multi_gpu --mixed_precision="fp16" ./sdxl_train.py \
  --poison_config_fp=$4 \
  --mixed_precision=fp16 \
  --use_8bit_adam \
  --xformers \
  --gradient_checkpointing \
  --log_with=wandb \
  --sample_at_first \
  --resolution=1024 \
  --sample_sampler=euler \
  --ddp_timeout=36000