# Generative Flow Model for Class Guided Video Synthesis
**Conditional OT Flow Matching** for class-conditioned video generation with **factorized space–time attention** for efficient temporal consistency.  
DDP-ready training + data pipeline + one-command sampling and training.



## Results:

**Sampling settings:** 
320×320 · 3s · 24fps · solver=Heun · NFE=40 · CFG_scale= 4.0 · checkpoint=100k training steps · model weights = MSE (condOT Flow Matching)

> Click any GIF to watch the full-quality Videos at 24fps.

> Previews are downsampled to 12fps , given git constraints.

> Demo uses a 100k-step checkpoint trained on a small curated set of short clips for proof-of-concept (Sintel + Dillama + Charged).



| Visuals| Warmth | tools | droid |
|---|---|---|---|
| [![visuals](assets/gifs/visuals.gif)](https://drive.google.com/file/d/1HhBnsss7t-Q3UKDOMBGw-yYgY8JbvtfV/view?usp=drive_link) | [![warmth](assets/gifs/warmth.gif)](https://drive.google.com/file/d/1Qi2ALIr_vTd32ckLhdgM83UXsTIp0K23/view?usp=drive_link) | [![tools](assets/gifs/tools.gif)](https://drive.google.com/file/d/1CeLPWRAGpzz4mc9zoSm9mhTlyEtOyWbh/view?usp=drive_link) | [![droid](assets/gifs/droid.gif)](https://drive.google.com/file/d/1Nh1xBrGMR0uhTAu23DDDCayhlmpRRzeu/view?usp=drive_link) |
| Sintel Scavenge | Sintel Motion | Sintel Conflict | Sintel Fire |
| [![Sintel Scavenge](assets/gifs/sintel%20scavenge.gif)](https://drive.google.com/file/d/1iFePHYWsmYPn79WRsLeCVmAw33_4isqC/view?usp=drive_link) | [![sintel motion](assets/gifs/sintel%20motion.gif)](https://drive.google.com/file/d/1PjNaNs05Z1Sn2mprPonjRk-YrHHkSMiQ/view?usp=drive_link) | [![Sintel Conflict](assets/gifs/sintel%20conflict.gif)](https://drive.google.com/file/d/1iDJJQMlDrE9_5eIao4eCQFdwKxNXrPnr/view?usp=drive_link) | [![Sintel Fire](assets/gifs/sintel%20fire.gif)](https://drive.google.com/file/d/1SEd4xqv6pCfmBcR63M62ON_6jWT1QvNv/view?usp=sharing) |

###### Conditional generations at 24fps over all 12 classes: https://drive.google.com/drive/folders/19gNgHTNiTFrXPcPosyOkNxC9rs18CiYB?usp=drive_link




## Setup:
git clone https://github.com/JiteshChavan/GenerativeFlow-Videos.git

cd GenerativeFlow-Videos

download data shard:
Download: [shard.zip](https://drive.google.com/file/d/1PTG88ff6UyATgdw8BqRn4jA4lLGhPjpg/view?usp=drive_link)

extract the zip in GenerativeFlow-Videos/flow/data

```bash
conda create -n flow python=3.10 -y
conda activate flow
conda install -c conda-forge ffmpeg -y
pip install -r req.txt
pip install -e .

# Starting training:
cd flow/scripts

# Run the DDP training script as 
bash grad_acc_flowS4.sh
```

## Inference:
Specify pre-trained checkpoint path in ```scripts/inference.sh``` and run the script for sampling videos.

```bash
SEED=051197
CKPT_PATH="checkpoint.pt"
BATCH_SIZE=2
VAE_FRAME_DECODE_BATCH=4

python -m flow.dnn.sample --inference-idx demo --ckpt-path $CKPT_PATH \
 --dnn-spec FlowField_S/4 \
 --temporal-res 72 \
 --batch-size $BATCH_SIZE \
 --vae-frame-decode-batch $VAE_FRAME_DECODE_BATCH \
 --solver heun \
 --sample-fps 24 \
 --learnable-pe \
 --use-temporal-attention \
 --seed  $SEED
```

## Pretrained checkpoint download coming soon!

## Method (Conditional OT Flow Matching + Factorized Space–Time Attention)

A marginal vector field $u^{\theta}(x,t,c)$ is approximated by a DNN, where $x$ is a video latent, $t\in[0,1]$ is continuous time, and $c$ is a class condition.  
Training uses **conditional OT flow matching**: sample an interpolation between data latents and noise, then regress the model’s vector field to the target conditional flow field with an MSE objective.  
At inference, we integrate the learned ODE from noise $\rightarrow$ data using a lightweight solver (Euler/Heun) with low NFE.


## Factorized Attention:
Factorized space–time attention applies spatial MHSA per frame and temporal MHSA per pixel enabling efficient temporal consistency. Cost: $O((T\cdot HW)^2) \rightarrow O(T(HW)^2 + HW\,T^2)$.
![NaiveAttention](/assets/FactorizedAttention.png)


To improve temporal consistency efficiently, the DNN uses **factorized space–time attention**: (1) spatial self-attention within each frame, then (2) temporal self-attention across frames per spatial location. This reduces attention cost from full space–time $O\big((T \cdot HW)^2\big)$ to $O\big(T \cdot (HW)^2 + HW \cdot T^2\big)$.

Conditioning (class + time) is injected via adaptive normalization (AdaLN-style modulation) and classifier-free guidance (CFG) is applied at sampling time.

Decoded frames are obtained by inverting the SD VAE, producing RGB videos at the target fps/resolution.

### TODO:
- [ ] scale to multimodal synthesis (Audio $\rightarrow$ Video)