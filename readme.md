# Generative Flow Model for Conditional Video Synthesis
**Conditional OT Flow Matching** for text-conditioned video generation with **factorized space–time attention** for efficient temporal consistency.  
DDP-ready training + data pipeline + one-command sampling and training.



## Results (GIF previews → MP4)

**Sampling settings:** 
320×320 · 3s · 24fps · solver=Heun · NFE=40 · CFG_scale= 4.0 · checkpoint=step100000

> Click any GIF to watch the full-quality GIFs at 24fps.
> The shocase has been downsampled to 12fps, given git constraints.

| Visuals| Warmth | tools | droid |
|---|---|---|---|
| [![visuals](assets/gifs/visuals.gif)](https://drive.google.com/file/d/1HhBnsss7t-Q3UKDOMBGw-yYgY8JbvtfV/view?usp=drive_link) | [![warmth](assets/gifs/warmth.gif)](https://drive.google.com/file/d/1Qi2ALIr_vTd32ckLhdgM83UXsTIp0K23/view?usp=drive_link) | [![tools](assets/gifs/tools.gif)](https://drive.google.com/file/d/1CeLPWRAGpzz4mc9zoSm9mhTlyEtOyWbh/view?usp=drive_link) | [![droid](assets/gifs/droid.gif)](https://drive.google.com/file/d/1Nh1xBrGMR0uhTAu23DDDCayhlmpRRzeu/view?usp=drive_link) |
| Sintel Scavenge | Sintel Motion | Sintel Conflict | Sintel Fire |
| [![Sintel Scavenge](assets/gifs/sintel%20scavenge.gif)](https://drive.google.com/file/d/1iFePHYWsmYPn79WRsLeCVmAw33_4isqC/view?usp=drive_link) | [![sintel motion](assets/gifs/sintel%20motion.gif)](https://drive.google.com/file/d/1PjNaNs05Z1Sn2mprPonjRk-YrHHkSMiQ/view?usp=drive_link) | [![Sintel Conflict](assets/gifs/sintel%20conflict.gif)](https://drive.google.com/file/d/1iDJJQMlDrE9_5eIao4eCQFdwKxNXrPnr/view?usp=drive_link) | [![Sintel Fire](assets/gifs/sintel%20fire.gif)](https://drive.google.com/file/d/1SEd4xqv6pCfmBcR63M62ON_6jWT1QvNv/view?usp=sharing) |




# Setup:
git clone https://github.com/JiteshChavan/GenerativeFlow-Videos.git

cd GenerativeFlow-Videos

download data shard:
https://drive.google.com/file/d/1KVQI1l-qH-D5YYJnA3pS9jJQCHK05Tsi/view?usp=drive_link

extract the zip in GenerativeFlow-Videos/flow/data


- conda create -n flow python=3.10 -y
- conda activate flow
- conda install -c conda-forge ffmpeg -y
- pip install -r req.txt
- pip install -e .

# Starting training:
cd flow/scripts

# Run script as 
bash grad_acc_flowS4.sh