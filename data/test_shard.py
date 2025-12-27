import io, glob
import torch
import webdataset as wds

pattern = "shards/000000.tar"
ds = wds.WebDataset(pattern).to_tuple("pt")

# grab first sample
pt_bytes, = next(iter(ds))
sample = torch.load(io.BytesIO(pt_bytes))

z = sample["z"]
label = sample["label_id"]

print(f"keys {sample.keys()}\n z: {z.shape, z.dtype} min/max{float(z.min())} {float(z.max())}")
print("label_id:", label)
print("clip_id:", sample.get("clip_id"))
print("label:", sample.get("label"))