import pandas as pd
from model import ResNet, BasicBlock
import torch
import joblib

if __name__ == "__main__":
    df = pd.read_parquet("rgz_umap_large.parquet")

    feat_cols = [f"feat_{i}" for i in range(512)]

    df = df[["feat_cols"]]
