from huggingface_hub import snapshot_download
import webdataset as wds
import numpy as np
import json
import io

def webdataset_cifar10() -> tuple[wds.WebDataset, wds.WebDataset, dict]:
    ds_path = snapshot_download(
        repo_id="Qualeafclover/webdataset-cifar10",
        repo_type="dataset",
    )
    with open(ds_path + "/info.json", "r") as f:
        info = json.load(f)
    train_ds_url = ds_path + "/train/cifar10-train-{000000..000007}.tar"
    train_ds = wds.WebDataset(train_ds_url, shardshuffle=False).with_length(info["train_len"])\
    .map(lambda data: {
        "label": json.loads(data["json"])["label"],
        "image": np.load(io.BytesIO(data["npy"])),
    })
    test_ds_url = ds_path + "/test/cifar10-test-{000000..000007}.tar"
    test_ds = wds.WebDataset(test_ds_url, shardshuffle=False).with_length(info["test_len"])\
    .map(lambda data: {
        "label": json.loads(data["json"])["label"],
        "image": np.load(io.BytesIO(data["npy"])),
    })
    return train_ds, test_ds, info
