import pandas as pd
import torch
import torchvision.transforms as T
import logging
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import joblib
from PIL import Image

from torch import Tensor
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from umap import UMAP
from byol.models import BYOL
from model import BasicBlock, ResNet

from byol.paths import Path_Handler
from byol.datamodules import RGZ108k
from byol.utilities import embed_dataset
from byol.models import BYOL


class RGZ(RGZ108k):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        remove_duplicates=True,
        cut_threshold=0.0,
        mb_cut=False,
    ):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            download,
            remove_duplicates=remove_duplicates,
            cut_threshold=cut_threshold,
            mb_cut=mb_cut,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index]
        mbf = self.mbflg[index]
        rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, index


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


# Get paths
print("Getting paths...")
path_handler = Path_Handler()
paths = path_handler._dict()

# Load in RGZ csv
print("Loading in RGZ data...")
csv_path = paths["rgz"] / "rgz_data.csv"
df_meta = pd.read_csv(csv_path)
df_meta = df_meta[["rgz_name", "radio.ra", "radio.dec", "radio.outermost_level"]]
df_meta = df_meta.rename(
    columns={"radio.dec": "dec", "radio.ra": "ra", "radio.outermost_level": "sigma"}
)

# Load transform
transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)
# Load in RGZ data
d_rgz = RGZ(
    paths["rgz"], train=True, transform=transform, remove_duplicates=True, cut_threshold=25, mb_cut=True
)


# Prepare hashmap for umap values
# y = {id: {"umap_x": None, "umap_y": None} for id in d_rgz.rgzid}

# Load model
# model_path = "byol.ckpt"
# checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

# byol = BYOL.load_from_checkpoint("byol.ckpt")
# byol.eval()
# encoder = byol.encoder.cuda()
# encoder.eval()

encoder = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], n_c=1, downscale=True, features=512).cuda()
encoder.load_state_dict(torch.load("encoder.pt"))
encoder.eval()


print("Encoding dataset...")
train_loader = DataLoader(d_rgz, 500, shuffle=False)
device = next(encoder.parameters()).device


feat_cols = [f"feat_{i}" for i in range(512)]

rgz_emb = []

for X, idx in tqdm(DataLoader(d_rgz, batch_size=500, shuffle=False)):
    B = X.shape[0]

    X_emb = encoder(X.cuda()).squeeze().detach().cpu().numpy().reshape((B, -1))
    rgz_emb.append(X_emb)

rgz_emb = np.concatenate(rgz_emb, axis=0)
print(f"Embedded batch shape: {X_emb.shape}, embedded dataset shape: {rgz_emb.shape}")

df = pd.DataFrame(columns=feat_cols, data=rgz_emb)
df["rgz_name"] = d_rgz.rgzid
df["size"] = d_rgz.sizes

# for X, idx in tqdm(DataLoader(d_rgz, batch_size=500, shuffle=False)):
#     names = d_rgz.rgzid[idx].tolist()
#     sizes = d_rgz.sizes[idx].tolist()

#     B = X.shape[0]

#     X_emb = encoder(X.cuda()).squeeze().detach().cpu().numpy().reshape((B, -1))

#     df_tmp = pd.DataFrame(
#         data=np.concatenate([names, sizes, X_emb], axis=1),
#         columns=["rgz_name", "size"] + feat_cols,
#     )

#     df = pd.concat([df, df_tmp], axis=0)

# Set dtypes
df = df.astype({feat_col: "float32" for feat_col in feat_cols})
df = df.astype({"rgz_name": "string"})

print(f"\n {len(df)} rows")
print(f"columns: {df.columns} \n")

print("Combining dataframe with meta data")
df = pd.merge(df, df_meta, on=["rgz_name"], how="inner")

print(f"\n {len(df)} rows")
print(f"columns: {df.columns} \n")
print(df.head())

# Save dataframe
print("Saving dataframe...")
df.to_parquet("embedded_rgz.parquet", index=False)


# Plot pca to check
# features = torch.tensor(df[feat_cols].values)
# features_test = encoder(next(iter(DataLoader(d_rgz, batch_size=len(d_rgz), shuffle=False)))).squeeze()
# print(torch.count_nonzero(features - features_test))
