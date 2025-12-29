import io
import random
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader


def _decode_pt(sample):
    return torch.load(io.BytesIO(sample["pt"]), map_location="cpu")


def _seed_worker(base_seed: int, rank: int, epoch: int):
    """Deterministic per-(epoch, rank, worker_id) RNG for python/numpy/torch."""
    def worker_init_fn(worker_id: int):
        s = int(base_seed) + 10_000 * int(epoch) + 1_000 * int(rank) + int(worker_id)
        random.seed(s)
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    return worker_init_fn


def build_wds_loader(
    shards,
    batch_size: int,
    num_workers: int,
    *,
    is_train: bool = True,
    epoch: int = 0,
    seed: int = 0,
    T_train: int = 48,
    stride: int = 1,
    hflip_p: float = 0.5,
    time_reverse_p: float = 0.0,
    shuffle_buf: int = 128,
    shard_shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    limit_samples: int = 0,          # <--- ADD THIS
):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0

    # If we're limiting samples, don't shardshuffle or you'll get a different subset each time.
    if limit_samples > 0:
        shard_shuffle = False

    dataset = wds.WebDataset(shards, shardshuffle=shard_shuffle)

    dataset = dataset.split_by_node() if hasattr(dataset, "split_by_node") else dataset.compose(wds.split_by_node)
    dataset = dataset.split_by_worker() if hasattr(dataset, "split_by_worker") else dataset.compose(wds.split_by_worker)

    # Deterministic order changes with epoch (if you want).
    if is_train:
        dataset = dataset.compose(wds.detshuffle(shuffle_buf, seed=seed + epoch))
        dataset = dataset.shuffle(shuffle_buf)

    # ---- LIMIT + REPEAT (DDP-safe) ----
    if limit_samples > 0:
        # Important: take AFTER split_by_node/worker, so each rank/worker sees a consistent subset.
        dataset = dataset.slice(limit_samples)   # first N samples in the stream
        dataset = dataset.repeat()               # infinite stream
    # ----------------------------------

    dataset = dataset.map(_decode_pt)

    def augment(rec):
        z = rec["z"]
        T_full = z.shape[0]

        if stride > 1:
            z = z[::stride]
            T_full = z.shape[0]

        if T_train < T_full:
            s = torch.randint(0, T_full - T_train + 1, (1,)).item()
            z = z[s:s + T_train]

        if hflip_p > 0 and torch.rand(1).item() < hflip_p:
            z = torch.flip(z, dims=[-1])

        if time_reverse_p > 0 and torch.rand(1).item() < time_reverse_p:
            z = torch.flip(z, dims=[0])

        rec["z"] = z
        return rec

    dataset = dataset.map(augment)

    def collate(batch):
        z = torch.stack([b["z"] for b in batch], dim=0)
        y = torch.tensor([int(b["label_id"]) for b in batch], dtype=torch.long)
        return {"z": z, "label_id": y}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and (num_workers > 0),
        worker_init_fn=_seed_worker(seed, rank, epoch),
        collate_fn=collate,
    )
    return loader
