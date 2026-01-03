# :)
GLOBAL_SEED=051197




EXP="FlowS4"

NUM_GPUS=1
BATCH_SIZE=16
TRAIN_STEPS=100000
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))



export WANDB_API_KEY="<INSERT YOU WANDB KEY>"

torchrun --standalone --nproc_per_node=$NUM_GPUS -m flow.dnn.train --exp $EXP --data-dir ../data/ \
  --dnn-spec FlowField_S/4 \
  --num-classes 12 \
  --temporal-res 72\
  --use-temporal-attention \
  --learnable-pe \
  --batch-size $BATCH_SIZE \
  --sample-bs $BATCH_SIZE \
  --lr 3e-4 \
  --train-steps $TRAIN_STEPS \
  --log-every 10 \
  --sample-every 500 \
  --ckpt-every 20000 \
  --save-content-every 10000 \
  --vae-frame-decode-batch 24 \
  --global-seed $GLOBAL_SEED \
  --sampler heun \
  --torch-compile
