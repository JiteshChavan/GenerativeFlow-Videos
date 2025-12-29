# decode_48.py
import argparse, io, os, tarfile
import torch
import cv2
from diffusers import AutoencoderKL

@torch.no_grad()
def decode_first_n_frames_to_mp4(z_T4hw_cpu, out_path, vae_name="ema", n_frames=48, fps=24, chunk=8):
    """
    z_T4hw_cpu: torch.Tensor [T,4,h,w] (cpu, fp16/fp32 ok)
    Writes an mp4 of the first n_frames decoded frames.
    """
    assert z_T4hw_cpu.dim() == 4 and z_T4hw_cpu.shape[1] == 4, f"expected [T,4,h,w], got {tuple(z_T4hw_cpu.shape)}"
    T = z_T4hw_cpu.shape[0]
    n = min(int(n_frames), int(T))

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to("cuda").eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # determine output spatial size from latent size
    h8, w8 = z_T4hw_cpu.shape[-2], z_T4hw_cpu.shape[-1]
    H, W = h8 * 8, w8 * 8

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))
    assert vw.isOpened(), f"cv2.VideoWriter failed to open: {out_path}"

    for s in range(0, n, chunk):
        z = z_T4hw_cpu[s:s+chunk].to("cuda", dtype=torch.float16, non_blocking=True)
        z = z / vae.config.scaling_factor

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            x = vae.decode(z).sample  # [k,3,H,W] in [-1,1]

        x = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]
        x = (x * 255).round().to(torch.uint8)  # [k,3,H,W]
        x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [k,H,W,3] RGB

        for i in range(x.shape[0]):
            bgr = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
            vw.write(bgr)

        del z, x
        torch.cuda.empty_cache()  # optional but helps on 8GB

    vw.release()
    print(f"wrote: {out_path}  (decoded {n}/{T} frames)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="path to a .tar shard, e.g. shards/000000.tar")
    ap.add_argument("--idx", type=int, default=0, help="which sample in the tar (0-based)")
    ap.add_argument("--out", default="decoded_48.mp4", help="output mp4 path")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--vae", choices=["ema", "mse"], default="ema")
    ap.add_argument("--n", type=int, default=72, help="number of frames to decode")
    ap.add_argument("--chunk", type=int, default=8, help="decode chunk size (smaller = safer)")
    args = ap.parse_args()

    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pt")]
        if not members:
            raise RuntimeError(f"no .pt found in {args.tar}")
        if args.idx < 0 or args.idx >= len(members):
            raise IndexError(f"--idx {args.idx} out of range (0..{len(members)-1})")
        m = members[args.idx]
        buf = tf.extractfile(m).read()
        obj = torch.load(io.BytesIO(buf), map_location="cpu")
        z = obj["z"]  # [T,4,h,w]

    decode_first_n_frames_to_mp4(
        z_T4hw_cpu=z,
        out_path=args.out,
        vae_name=args.vae,
        n_frames=args.n,
        fps=args.fps,
        chunk=args.chunk,
    )

if __name__ == "__main__":
    main()
