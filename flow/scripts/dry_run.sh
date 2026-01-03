# :)
GLOBAL_SEED=051197




EXP="debug"

NUM_GPUS=1
BATCH_SIZE=2
TRAIN_STEPS=10000
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
  --log-every 2 \
  --sample-every 200 \
  --ckpt-every 10000 \
  --save-content-every 5000 \
  --vae-frame-decode-batch 4 \
  --global-seed $GLOBAL_SEED \
  --sampler heun \
  #--torch-compile
