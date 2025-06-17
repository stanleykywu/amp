export CUDA_VISIBLE_DEVICES=$1

accelerate launch --main_process_port=$3 --num_processes=$2 --mixed_precision="fp16" ./sd21_train.py \
  --poison_config_fp=$4 \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --resume_from_checkpoint=latest
  