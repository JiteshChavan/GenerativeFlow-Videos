import io, tarfile, torch
from diffusers import AutoencoderKL

device = "cuda"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

@torch.no_grad()
def decode_latents_chunked(z, chunk=8):
    """
    z: [T,4,h,w] (cpu or gpu, fp16 ok)
    returns x: [T,3,H,W] in [0,1] on cpu
    """
    T = z.shape[0]
    outs = []
    for s in range(0, T, chunk):
        zz = z[s:s+chunk].to(device, dtype=torch.float16, non_blocking=True)
        zz = zz / vae.config.scaling_factor
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = vae.decode(zz).sample  # [-1,1]
        x = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]
        outs.append(x.cpu())
        del zz, x
        torch.cuda.empty_cache()  # optional; remove if it slows you down
    return torch.cat(outs, dim=0)

tar_path = "shards/000000.tar"
with tarfile.open(tar_path, "r") as tf:
    members = [m for m in tf.getmembers() if m.name.endswith(".pt")]
    m = members[0]
    buf = tf.extractfile(m).read()
    obj = torch.load(io.BytesIO(buf), map_location="cpu")
    z = obj["z"]  # [T,4,h,w]
    x = decode_latents_chunked(z, chunk=4)  # try 4, 8, 16 depending on VRAM
    print(x.shape, x.min().item(), x.max().item(), x.dtype)
