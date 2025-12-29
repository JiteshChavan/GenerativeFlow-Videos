import json
import torch
from loader import build_wds_loader
import glob
import tarfile


def count_unique_keys_in_tar(tar_path: str) -> int:
    keys = set()
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)          
            key, _ext = os.path.splitext(base)       
            keys.add(key)
    return len(keys)

labels = json.load(open("clips/labels.json"))
id_to_name = {v: k for k, v in labels.items()}
for i in sorted(id_to_name):
    print(i, id_to_name[i])

shards = sorted(glob.glob("shards/*.tar"))
loader = build_wds_loader(
    shards=shards,  # or "shards_latent/*.tar"
    batch_size=2,
    num_workers=0,
    is_train=False,        
    epoch=0,
    seed=0,
    T_train=48,            
    stride=1,
    hflip_p=0.0,
    time_reverse_p=0.0,
    shuffle_buf=64,        
    shard_shuffle=False,
    drop_last=False,
)

b = next(iter(loader))

print("z:", b["z"].shape, b["z"].dtype)
print("label_id:", b["label_id"], b["label_id"].dtype)


print("z min/max:", b["z"].min().item(), b["z"].max().item())
print("labels in batch:", torch.unique(b["label_id"]).tolist())

print("done")
