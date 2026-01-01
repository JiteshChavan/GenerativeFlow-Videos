import io, json
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class Augmentation:
    def __init__(self, p_time_reverse: float=0.0, p_hflip=0.0):
        self.p_time_reverse = p_time_reverse
        self.p_hflip = p_hflip
    
    def __call__(self, z: torch.Tensor)->torch.Tensor:
        # Z (T, 4, H, W)
        if self.p_time_reverse > 0 and torch.rand(()) < self.p_time_reverse:
            z = torch.flip(z, dims=[0])
        if self.p_hflip > 0 and torch.rand(()) < self.p_hflip:
            z = torch.flip(z, dims=[3])
        return z


class VideoLatentLMDB(Dataset):
    """
    Expects each key '000000123' -> torch.save(dict) containing at least:
      - 'z' : fp16 CPU tensor [T,4,H,W]
      - 'label_id' : int
      - '__len__' : ascii int
      - '__metadata__' : json (optional)
    """

    def __init__(self, lmdb_path: str, transform=None, return_meta: bool = False):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.return_meta = return_meta
        self.env = None
    
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            self._len = int(txn.get(b"__len__").decode("ascii"))
            meta_raw = txn.get(b"__metadata__")
            self._meta = json.loads(meta_raw.decode("utf-8")) if meta_raw else {}
        env.close()
    
    @property
    def meta(self):
        return self._meta
    
    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx: int):
        self._init_env()
        key = f"{idx:09d}".encode("ascii")
        with self.env.begin() as txn:
            raw = txn.get(key)
            if raw is None:
                raise IndexError(idx)
        
        sample = torch.load(io.BytesIO(raw), map_location="cpu")
        z = sample["z"]
        y = int (sample["label_id"])

        if self.transform is not None:
            z = self.transform(z)
        
        out = {"z": z , "label_id": y}
        if self.return_meta:
            for k in ("label", "clip_id", "clip_relpath"):
                if k in sample:
                    out[k] = sample[k]
            
        return out

def collate_latents(batch):
    z = torch.stack([b["z"] for b in batch], dim=0) # (B, T, 4, H, W)
    y = torch.tensor([b["label_id"] for b in batch], dtype= torch.long) #(B)
    out = {"z": z, "label_id": y}

    for k in ("label", "clip_id", "clip_relpath"):
        if k in batch[0].keys():
            out[k] = [b.get(k) for b in batch]
    return out
    
def make_lmdb_dataloader(
    lmdb_path: str,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool=True,
    p_time_reverse: float = 0.0,
    p_hflip: float = 0.5,
    return_meta: bool = False,
):
    transform = Augmentation(p_time_reverse, p_hflip)
    ds = VideoLatentLMDB(lmdb_path, transform=transform, return_meta=return_meta)

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=drop_last)
        shuffle = False # sampler handles shuffling not loader
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate_latents,
        drop_last=drop_last if sampler is None else False, # sampler handles drop last
    )
    return ds, dl, sampler
    

