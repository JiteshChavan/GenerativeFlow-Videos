# Generative Flow Model for Conditional Video Synthesis
**Conditional OT Flow Matching** for text-conditioned video generation with **factorized space–time attention** for efficient temporal consistency.  
DDP-ready training + data pipeline + one-command sampling and training.



## Results (GIF previews → MP4)

**Demo settings:** 
320×320 · 3s · 24fps · solver=Heun · NFE=40 · CFG_scale= 4.0 · checkpoint=step100000

> Click any GIF to watch the full-quality MP4.
| | | | |
|---|---|---|---|
|[![visuals](assets/flow_inference/visuals.gif)](https://drive.google.com/file/d/1GPJ7-hTaxLxhO2Wur0zg4yw4uK9jag5-/view?usp=sharing) | [![warmth](assets/gifs/warmth.gif)](https://drive.google.com/file/d/1qRHsubEr2ixMOj34tKwJxTFzdcqlYL3w/view?usp=drive_link) | [![tools](assets/gifs/tools.gif)](https://drive.google.com/file/d/1SkKyNMd70ICiEoav_n9Gnvg0AQNt56ao/view?usp=drive_link) |  [![droid](assets/gifs/droid.gif)](https://drive.google.com/file/d/1IWg9pgwp4g7_i8TNO4B5LvTYj3H4gxSs/view?usp=drive_link) |  



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