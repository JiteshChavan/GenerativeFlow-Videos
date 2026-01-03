import shutil
import random
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm
import numpy as np


import cv2

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from diffusers.models import AutoencoderKL
from .dnn import create_dnn
from .flowMatching import FlowMatching, FlowSampler

from flow.data.lmdb_dataset import make_lmdb_dataloader

import inspect
import wandb


import subprocess

def get_ffmpeg_exe() -> str:
    #exe = shutil.which("ffmpeg")
    #if exe is not None:
    #    return exe

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg binary not found. Install it via:\n"
            "  conda install -c conda-forge ffmpeg -y\n"
            "or:\n"
            "  pip install imageio-ffmpeg\n"
        ) from e

def reencode_h264(in_path: str, out_path: str):
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)

def save_mp4(samples, path, fps=24):
    # samples (T, 3, H, W) in [0,1]
    samples = (samples.clamp(0,1) * 255).round().to(torch.uint8)
    samples = samples.permute(0, 2, 3, 1).contiguous().cpu().numpy() # (T, H, W, 3) RGB
    T, H, W, C = samples.shape

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for i in range(T):
        bgr = cv2.cvtColor(samples[i], cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    
    vw.release()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file.
    """

    if dist.get_rank() == 0: # master process
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger




def adjust_learning_rate(optimizer, step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < args.warmup_steps:
        lr = args.lr * step / args.warmup_steps
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (step - args.warmup_steps) / (args.train_steps - args.warmup_steps))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def configure_optimizers (DNN, weight_decay, learning_rate, device):
        # start with all the candidate parameters (that require grad)
        param_dict = {pn : p for pn, p in DNN.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all tensors in matmul + embeddings decay, all biases and layer norms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0}
        ]
        num_decay_params = sum (p.numel () for p in decay_params)
        num_nodecay_params = sum (p.numel () for p in nodecay_params)
        print (f"num decayed parameter tensors : {len(decay_params)} with {num_decay_params:,} parameters")
        print (f"num non-decayed parameter tensors : {len(nodecay_params)} with  {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        print (f"using fused AdamW: {use_fused}")
        # kernel fusion for AdamW update instead of iterating over all the tensors to step which is a lot slower.
        optimizer = torch.optim.AdamW (optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def ddp_setup(global_seed: int):
    """
    DDP init for torchrun
    returns: rank, local_rank, world_size, device
    """

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int (os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

    seed = global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return rank, local_rank, world_size, device




def main(args):
    """
    Approximate a flow field instrumenting a DNN
    """

    assert torch.cuda.is_available(), f"Training requires at least one GPU currently."

    rank, local_rank, world_size, device = ddp_setup(args.global_seed)


    # Setup an experiment folder:
    experiment_index = args.exp
    experiment_dir = f"{args.results_dir}/{experiment_index}" # create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints" # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        mode = "online" if args.use_wandb else "disabled"
        wandb.init(
            project = "FlowVideo",
            config=vars(args),
            name=experiment_index,
            mode=mode
        )
        print ("W&B run id: ", wandb.run.id)
        print("W&B url:", wandb.run.url)
    else:
        logger = create_logger(None)
    
    # Create model + dnn
    assert args.spatial_res % 8 == 0, "Resolution must be divisible by 8 (for the VAE)"
    assert args.spatial_res // 8 == args.latent_res

    DNN = create_dnn(args)

    if args.torch_compile:
        DNN = torch.compile(DNN, mode="max-autotune", fullgraph=False)

    DNN = DDP(DNN.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    flow = FlowMatching(DNN, t_sampler=args.t_sampler)
    sampler = FlowSampler(DNN.module, args.sampler)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}",torch_dtype=torch.float16).eval().to("cpu")
    vae.requires_grad_(False)

    logger.info(f"Model :{args.dnn_spec} Parameter count : {sum(p.numel() for p in DNN.parameters()):,}")

    optimizer = configure_optimizers(DNN, weight_decay=0.1, learning_rate=args.lr, device=device)

    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint_file = os.path.join(checkpoint_dir, "content.pt")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f"cuda:{device}"), weights_only=False)
        init_epoch = checkpoint["epoch"]
        init_step = checkpoint["step"] + 1
        DNN.module.load_state_dict(checkpoint["dnn"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for g in optimizer.param_groups:
            g["lr"] = args.lr
        
        logger.info(f"> resume checkpoint (epoch :{init_epoch} step: {init_step})")
        del checkpoint
        gc.collect()
    else:
        init_epoch = 0
        init_step = 0
        

    # DATA LAODER
    data_dir = os.path.join(args.data_dir, "lmdb", "train.lmdb")
    assert os.path.exists(data_dir), f"specified datadir :{data_dir} does not exist."

    batch_size = args.batch_size

    epoch = init_epoch
    current_step = init_step
    total_steps = args.train_steps

    
    dataset, train_loader, data_sampler = make_lmdb_dataloader(
        lmdb_path=data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        is_distributed=(world_size > 1),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        p_hflip=args.p_hflip,
        p_time_reverse=args.p_time_reverse,
        return_meta=False,
    )


    logger.info(f"Found {len(dataset)} examples.")

    
    if data_sampler is not None:
        data_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)

    def next_batch():
        nonlocal data_iter, epoch, train_loader

        try:
            return next(data_iter)
        except StopIteration:
            epoch += 1
            if data_sampler is not None:
                data_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            return next(data_iter)

    
    # TRAINING!!
    DNN.train()

    log_steps = 0
    running_loss = 0
    running_grad_norm = 0.0
    start_time = time()

    # samples for inference
    latent_res = args.latent_res
    spatial_res = args.spatial_res
    latent_channels = args.latent_channels
    sample_bs = args.sample_bs
    sample_frame_count = args.temporal_res
    if rank == 0:
        zs = torch.randn(sample_bs, sample_frame_count, latent_channels, latent_res, latent_res, device=device)
        c = torch.arange(0, args.num_classes, step=1, dtype=torch.long, device=device)

        if sample_bs <= args.num_classes:
            c = c[:sample_bs]
        else:
            repeats = (sample_bs + args.num_classes - 1) // args.num_classes
            c = c.repeat(repeats)[:sample_bs]


    total_steps = args.train_steps
    current_step = init_step
    logger.info(f"Training for {total_steps} steps...")


    for current_step in range(init_step, total_steps):
        batch = next_batch()
        x = batch["z"].to(device, non_blocking=True)
        y = batch["label_id"].to(device, non_blocking=True)
        #adjust_learning_rate(optimizer, current_step, args)


        before_forward = torch.cuda.memory_allocated(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = flow.training_step(x, y)
        after_forward = torch.cuda.memory_allocated(device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(DNN.parameters(), args.max_grad_norm)

        optimizer.step()
        after_backward = torch.cuda.memory_allocated(device)

        # Log loss values:
        running_loss += loss.item()
        running_grad_norm += grad_norm.item()
        log_steps += 1

        # control switches for logging
        will_log_samples = (rank == 0) and (current_step % args.sample_every == 0)
        
        if current_step % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size

            # Reduce grad norm history over all processes:
            avg_grad_norm = torch.tensor(running_grad_norm / log_steps, device = device)
            dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
            avg_grad_norm = avg_grad_norm.item() / world_size

            # log
            if rank == 0:
                wandb.log(
                    {
                        "train_loss" : avg_loss,
                        "train_steps_per_sec" : steps_per_sec,
                        "gpu_mem/before_fwd_gb" : before_forward/1e9,
                        "gpu_mem/after_fwd" : after_forward/1e9,
                        "gpu_mem/after_backward" : after_backward/1e9,
                        "grad/avg_norm" : avg_grad_norm,
                    },
                    step=current_step, # drives x-axis
                    commit=not(will_log_samples)
                )

                logger.info(
                    f"(epoch={epoch}/step={current_step}) Train loss: {avg_loss:.4f},"
                    f"Train Steps/Sec: {steps_per_sec:.2f},"
                    f"GPU Mem before forward: {before_forward/10**9:.2f}Gb,"
                    f"GPU Mem after forward: {after_forward/10**9:.2f}Gb,"
                    f"GPU Mem after backward: {after_backward/10**9:.2f}Gb,"
                    f"Avg Grad Norm: {avg_grad_norm:.4f}"
                )
            # Reset monitoring vairables
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        # if not args.no_lr_decay:
        #     scheduler.step()

        if rank == 0:
            # latest checkpoint
            if current_step % args.save_content_every == 0 and current_step != 0:
                logger.info("Saving content.")
                content = {
                    "epoch": epoch,
                    "step": current_step,
                    "args": args,
                    "dnn": DNN.module.state_dict(),
                    "optimizer": optimizer.state_dict(),

                    "data_loader_state": {
                        "epoch": epoch,
                    }
                }

                torch.save(content, os.path.join(checkpoint_dir, f"content.pt"))

            # checkpoint
            if current_step % args.ckpt_every == 0 and current_step > 0:
                logger.info ("Saving checkpoint.")
                checkpoint = {
                    "epoch" : epoch,
                    "step" : current_step,
                    "dnn" : DNN.module.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch{epoch:07d}_step{current_step}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}.")

        if rank == 0 and current_step % args.sample_every == 0:
            optimizer.zero_grad(set_to_none=True)
            logger.info(f"Generating samples...")
            DNN.eval()
            vae.eval()
            vae.to(device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    samples = sampler.sample(zs, c, steps=40, cfg_scale=4.0) # (B, T, 4, H, W)
                
                B, T, C, H, W = samples.shape
                samples = samples.reshape(B*T, C, H, W)
                samples = samples / vae.config.scaling_factor

                decoded = []
                chunk = args.vae_frame_decode_batch
                for s in range(0, samples.shape[0], chunk):
                    x = samples[s:s+chunk]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        img = vae.decode(x).sample # in [-1,1]
                    img = (img.clamp(-1,1) + 1) * 0.5 # in [0, 1]
                    decoded.append(img.float().cpu())
                    del x, img
                
                del samples
                torch.cuda.empty_cache()

                samples = torch.cat(decoded, dim=0) # (B * T, 3, H, W)
                samples = samples.reshape(B, T, 3, samples.shape[-2], samples.shape[-1])

            vae.to("cpu")
            del decoded
            torch.cuda.empty_cache()
            # Save and display videos
            videos = []
            for b in range(B):
                cls = int(c[b].item())
                tmp = os.path.join(sample_dir, f"class-{cls}-step{current_step}-{b}-tmp.mp4")
                out = os.path.join(sample_dir, f"class-{cls}-step{current_step}-{b}.mp4")

                save_mp4(samples[b], tmp, fps=args.sample_fps)
                reencode_h264(tmp, out)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                videos.append(wandb.Video(out, fps=args.sample_fps, format="mp4"))
            wandb.log({"samples": videos}, step=current_step, commit=will_log_samples)
            DNN.train()
            del samples
            gc.collect()
            torch.cuda.synchronize()
    
    DNN.eval()
    logger.info("Done!")
    dist.destroy_process_group()



def none_or_str(value):
    if value == "None":
        return None
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, required=True, help="name of the experiment")
    parser.add_argument("--dnn-spec", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)

    parser.add_argument("--results-dir", type=str, default="results", help="results directory")
    

    parser.add_argument("--global-seed", type=int, help="rng seed", required=True)
    parser.add_argument("--num-classes", type=int, default=12, help="rng seed")
    parser.add_argument("--use-wandb", action="store_true")

    parser.add_argument("--spatial-res", type=int, default=320)
    parser.add_argument("--latent-res", type=int, default=40)
    parser.add_argument("--temporal-res", type=int, default=48)
    parser.add_argument("--latent-channels", type=int, default=4)

    parser.add_argument("--use-temporal-attention", action="store_true")
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--label-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-bs", type=int, default=2)
    parser.add_argument("--sample-fps", type=int, default=24)
    parser.add_argument("--vae-frame-decode-batch", type=int, default=24)

    parser.add_argument("--t-sampler", type=str, default="uniform", choices=["uniform", "logit_normal"])
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"])

    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)    
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=12)
    

    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--steps-per-reshuffle", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--save-content-every", type=int, default=1000)


    parser.add_argument("--model-ckpt", type=str, default="")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)

    

    parser.add_argument("--p-hflip", type=float, default=0.5)
    parser.add_argument("--p-time-reverse", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
    





