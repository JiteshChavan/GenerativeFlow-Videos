import os, io, json, hashlib
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

def stable_clip_id(relpath: str)->str:
    return hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:15]

def list_videos_by_class(root: Path):
    classes = []
    items = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            classes.append(d.name)
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in VIDEO_EXTS:
                    rel = p.relative_to(root).as_posix()
                    items.append((rel, d.name))
    return classes, items

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap

def read_window_seek(cap, start: int, T: int):
    #  Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frames = []
    for _ in range(T):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    return frames

def read_random_window(path: str, T: int, rng: np.random.Generator):
    cap = open_video(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        # fallback: decode sequentially
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len (frames) == 0:
            raise RuntimeError(f"no frames decoded: {path}")
        n = len(frames)
        if n < T:
            frames = frames + [frames[-1] * (T - n)]
        s = int (rng.integers(0, max(1, len(frames) - T + 1)))
        return frames[s:s+T]
    
    # choose start
    if n < T:
        s = 0
    else:
        s = int (rng.integers(0, n-T+1))
    
    frames = read_window_seek(cap, s, T)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read: {path}")
    
    # pad if short
    if len(frames) < T:
        frames = frames + [frames[-1]] * (T - len(frames))
    return frames

def random_crop(frames, crop:int, rng: np.random.Generator):
    if crop is None:
        return frames
    
    h, w = frames[0].shape[:2]
    if h < crop or w < crop:
        # if too small, center pad by resizing up
        frames = [cv2.resize(fr, (max(w, crop), max(h, crop)), interpolation=cv2.INTER_AREA) for fr in frames]
        h, w = frames[0].shape[:2]
    y0 = int (rng.integers(0, h - crop + 1))
    x0 = int (rng.integers(0, w - crop + 1))
    return [fr[y0:y0+crop, x0:x0+crop] for fr in frames]

def frames_to_vae_input(frames_bgr):
    # to torch float (T, 3, H, W) in [-1, 1]
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames_bgr]
    arr = np.stack(rgb, 0) #(T, H, W, 3) uint8
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
    x = x * 2.0 - 1.0
    return x

def estimate_map_size (num_samples: int, bytes_per_sample: int, safety:float=1.3)->int:
    return int(num_samples * bytes_per_sample * safety)

def main (
    clips_root: str = "./clips",
    lmdb_path: str = "./lmdb/train.lmdb",
    T_train: int = 72,
    crop: int = 320,
    vae_name = "ema", # "ema" or "mse"
    seed: int = 51197,
    commit_every: int = 256,
    vae_batch_decode: int = 72,
):
    root = Path(clips_root)
    classes, items = list_videos_by_class(root)
    assert len(items) > 0, f"no videos found under :{root}"

    label2id = { c:i for i,c in enumerate(classes)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device, dtype=dtype)
    vae.eval()

    # z is (T, 4, (crop/8), (crop/8)) fp16 -> T*4*(crop/8)^2 *2 bytes
    hz = crop // 8 # vae downscale factor
    bytes_per = T_train * 4 * hz * hz * 2
    map_size = estimate_map_size(len(items), bytes_per_sample=bytes_per, safety=1.6)
    map_size = max(map_size, 512 << 20) # atleast 512MB

    p = Path(lmdb_path)
    if p.exists():
        import shutil
        shutil.rmtree(p)

    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(
    lmdb_path,
    map_size=map_size,
    subdir=True,   # lmdb_path is a directory like ".../train.lmdb"
    readonly=False,
    create=True,
    lock=True,
)

    rng = np.random.default_rng(seed)
    txn = env.begin(write=True)
    meta = {
        "clips_root" : str(root.resolve()),
        "T_train" : T_train,
        "crop" : crop,
        "vae" : f"stabilityai/sd-vae-ft-{vae_name}",
        "dtype": "float16",
        "latent_shape": [T_train, 4, crop // 8, crop // 8],
        "classes" : classes,
        "label2id" : label2id,
    }

    n_written = 0
    print(f"[LMDB] writing to : {lmdb_path}")
    print(f"[LMDB] samples: {len(items)}, mapsize={map_size/1e9:.2f} GB, device={device}")

    B = args.vae_batch_decode
    assert T_train % B == 0, f"temporal res {T_train} is not divisible by vae decode batch size : {B}"
    for idx, (relpath, label) in enumerate(tqdm(items)):
        abspath = (root / relpath).as_posix()
        try:
            frames = read_random_window(abspath, T_train, rng)
            frames = random_crop(frames, crop, rng)
            x = frames_to_vae_input(frames) #(T, 3, H, W)
            batches = x.shape[0] // B
            decoded = []
            with torch.no_grad():
                for batch in range(batches):
                    samples = x[batch* B: (batch + 1) * B].to(device=device, dtype=vae.dtype)
                    samples = vae.encode(samples).latent_dist.sample() * vae.config.scaling_factor
                    samples = samples.to(torch.float16).cpu()
                    decoded.append(samples)
            z = torch.cat(decoded, dim=0)

            payload = {
                "z" : z, # fp16 torch tensor on cpu
                "label_id": int(label2id[label]),
                "label" : label,
                "clip_relpath": relpath,
                "clip_id":stable_clip_id(relpath),
            }

            buf = io.BytesIO()
            torch.save(payload, buf) # assumes trusted reads
            key = f"{n_written:09d}".encode("ascii")
            txn.put(key, buf.getvalue())
            n_written += 1

            if (n_written % commit_every) == 0:
                txn.commit()
                txn = env.begin(write = True)
        
        except Exception as e:
            # skip bad clips
            print(f"[SKIP] {relpath}: {e}")
    
    txn.commit()
    with env.begin(write=True) as txn2:
        txn2.put(b"__len__", str(n_written).encode("ascii"))
        txn2.put(b"__metadata__", json.dumps(meta).encode("utf-8"))
    
    env.sync()
    env.close()
    print(f"[done] wrote {n_written} samples to {lmdb_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-root", type=str, default="./clips")
    parser.add_argument("--lmdb-path", type=str, default="./lmdb/train.lmdb")
    parser.add_argument("--T-train", type=int, default=72)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--vae-name", type=str, default="ema")
    parser.add_argument("--seed", type=int, default=51197)
    parser.add_argument("--commit-every", type=int, default=256)
    parser.add_argument("--vae-batch-decode", type=int, default=24)
    args = parser.parse_args()
    main(**vars(args))

