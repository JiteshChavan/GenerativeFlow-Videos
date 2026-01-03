import os
import argparse

import torch


from flow.dnn.dnn import create_dnn
from flow.dnn.flowMatching import FlowSampler
from diffusers.models import AutoencoderKL
import cv2
import gc
import subprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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


def main(args):
    assert torch.cuda.is_available(), f"Sampling requires GPU currently."
    assert os.path.exists(args.ckpt_path), f"specified ckpt :{args.ckpt_path} does not exist."

    inference_dir = args.inference_dir
    inference_idx = args.inference_idx
    inference_path = os.path.join(inference_dir, inference_idx)

    os.makedirs(inference_dir,exist_ok=True)
    os.makedirs(inference_path,exist_ok=True)

    ckpt_file = args.ckpt_path

    assert args.spatial_res % 8 == 0, f"Resolution must be divisible by 8 for the VAE."
    assert args.spatial_res // 8 == args.latent_res

    DNN = create_dnn(args)
    DNN = DNN.to('cuda')
    DNN.eval()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    flow_sampler = FlowSampler(DNN, sampler=args.solver)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}",torch_dtype=torch.float16).eval().to('cuda')

    checkpoint = torch.load(ckpt_file, map_location="cuda", weights_only=False)
    DNN.load_state_dict(checkpoint["dnn"])


    B = args.batch_size
    T = DNN.temporal_resolution
    H = DNN.spatial_resolution
    W = DNN.spatial_resolution
    C = DNN.in_channels
    
    if args.classes is None or len(args.classes) == 0:
        y = torch.randint(0, DNN.num_classes, (B,), dtype=torch.long).to('cuda')
    else:
        
        assert len(args.classes) == B, f"Mismatch between number of inference classes :{len(args.classes)} and specified batch size :{B}."
        y = torch.tensor(args.classes, dtype=torch.long, device='cuda')

    x0 = torch.randn(B, T, C, H, W).to('cuda')


    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            x1 = flow_sampler.sample(x0, y, steps=30, cfg_scale=4.0) # (B, T, C, H, W)
        
        samples = x1.reshape(B*T, C, H, W)
        samples = samples / vae.config.scaling_factor

        decoded = []
        
        b = args.vae_frame_decode_batch
        assert (B*T) % b == 0, f"flattened batch*frames : {B*T} should be divisble by vae_frame_decode_batch :{b}"
        batches = samples.shape[0] // b
        for batch in range(batches):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                frames = vae.decode(samples[batch*b:(batch + 1)*b]).sample # in [-1, 1]
            frames = (frames.clamp(-1, 1) + 1) * 0.5 # in [0, 1]
            decoded.append(frames.float().cpu())
            del frames
        
        del samples
        videos = torch.cat(decoded, dim=0) #(BT, 3, spatial_res, spatial_res)
        videos = videos.reshape(B, T, 3, videos.shape[-2], videos.shape[-1])
        assert videos.shape[0] == B and videos.shape[1] == T, videos.shape
    del decoded
    torch.cuda.empty_cache()

    # Save and display videos
    for b in range(B):
        cls = int(y[b].item())
        tmp = os.path.join(inference_path, f"{b:02d}-class-{cls}-tmp.mp4")
        out = os.path.join(inference_path, f"{b:02d}-class-{cls}.mp4")

        save_mp4(videos[b], tmp, fps=args.sample_fps)
        reencode_h264(tmp, out)
        try:
            os.remove(tmp)
        except OSError:
            pass
    
    gc.collect()
    torch.cuda.synchronize()
    print ("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--inference-dir", type=str, default="inferences")
    parser.add_argument("--inference-idx", type=str, default="sample_demo")
    parser.add_argument("--ckpt-path", type=str, required=True)


    parser.add_argument("--spatial-res", type=int, default=320)
    parser.add_argument("--latent-res", type=int, default=40)
    parser.add_argument("--temporal-res", type=int, default=72)

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--vae-frame-decode-batch", type=int, required=True)
    parser.add_argument("--solver", type=str, required=True, choices=["heun", "euler"])
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    parser.add_argument("--classes", type=int, nargs="*", default=None, help="[Optional] specify list of classes to sample from; has to be same size as specified batch size")

    parser.add_argument("--sample-fps", type=int, default=24)

    # dnn spec
    parser.add_argument("--dnn-spec", type=str, required=True, choices=["FlowField_XS/4", "FlowField_S/4", "FlowField_S/2", "FlowField_M/2"])
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--label-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--use-temporal-attention", action="store_true")

    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    main(args)



