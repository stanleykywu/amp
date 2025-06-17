export CUDA_VISIBLE_DEVICES=$1

accelerate launch --main_process_port=$3 --num_processes=$2 --multi_gpu --num_cpu_threads_per_process 1 ./flux_train.py \
  --poison_config_fp=$4 \
  --sdpa \
  --persistent_data_loader_workers \
  --max_data_loader_n_workers 2 \
  --save_precision bf16 \
  --mixed_precision bf16 \
  --highvram \
  --optimizer_type adafactor \
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
  --max_grad_norm 0.0 \
  --timestep_sampling shift \
  --discrete_flow_shift 3.1582 \
  --model_prediction_type raw \
  --guidance_scale 1.0 \
  --full_bf16 \
  --log_with wandb \
  --gradient_checkpointing \
  --ddp_timeout 2160000000

