import pandas as pd
from model import ResNet, BasicBlock
import torch
import joblib

if __name__ == "__main__":
    df = pd.read_parquet("umap.parquet")

    df = df[["rgz_name", "umap_x", "umap_y", "ra", "dec", "size"]]

    df = df.astype({col: "float32" for col in ["umap_x", "umap_y", "ra", "dec", "size"]})
    df = df.astype({"rgz_name": "string"})
    df.to_parquet("umap.parquet")
    print(df.head())

    # feat_cols = [f"feat_{i}" for i in range(512)]
    # print(f"Length: {len(df)} \n Columns: {df.columns}")

    # print(df.head())

    # float_cols = df.columns.values.tolist()
    # float_cols.remove("rgz_name")

    # df = df.astype({float_col: "float32" for float_col in float_cols})
    # df = df.astype({"rgz_name": "string"})

    # print(df["rgz_name"].dtype, df["feat_0"].dtype)

    # # print(df.head())

    # df = df[["rgz_name", "ra", "dec", "umap_x", "umap_y"]]

    # df.to_parquet("rgz_umap.parquet")

    # resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], n_c=1, downscale=True, features=512)
    # resnet.load_state_dict(torch.load("encoder.pt"))

    # reducer = joblib.load("reducer.joblib")
