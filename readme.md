

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