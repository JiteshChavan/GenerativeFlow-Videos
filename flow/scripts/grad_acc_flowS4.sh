# :)
GLOBAL_SEED=051197

EXP="FlowS4"

NUM_GPUS=1
GRAD_ACC_STEPS=2
BATCH_SIZE=24
TRAIN_STEPS=100000

export WANDB_API_KEY="<INSERT YOU WANDB KEY>"
torchrun --standalone --nproc_per_node=$NUM_GPUS -m flow.dnn.grad_acc_train --exp $EXP --data-dir flow/data/ \
  --dnn-spec FlowField_S/4 \
  --num-classes 12 \
  --temporal-res 72\
  --use-temporal-attention \
  --learnable-pe \
  --batch-size $BATCH_SIZE \
  --sample-bs $BATCH_SIZE \
  --lr 3e-4 \
  --train-steps $TRAIN_STEPS \
  --log-every 2 \
  --sample-every 500 \
  --ckpt-every 10000 \
  --save-content-every 5000 \
  --vae-frame-decode-batch 24 \
  --global-seed $GLOBAL_SEED \
  --sampler heun \
  --grad-acc-steps $GRAD_ACC_STEPS\
  --use-wandb \
  #--torch-compile \
