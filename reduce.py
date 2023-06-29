import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from collections import Counter
from sklearn.decomposition import PCA
from umap import UMAP


PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 50
UMAP_MIN_DIST = 0.0005
METRIC = "cosine"
seed = 69


df = pd.read_parquet("embedded_rgz.parquet")
features = df[[f"feat_{i}" for i in range(512)]].values

print("Fitting PCA...")
pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
pca.fit(features)
joblib.dump(pca, "pca.joblib")

pca_cols = [f"pca_{i}" for i in range(PCA_COMPONENTS)]
df_pca = pd.DataFrame(data=pca.transform(features), columns=pca_cols)
df = pd.concat([df, df_pca], axis=1)
df[pca_cols + ["rgz_name", "size", "ra", "dec"]].to_parquet("pca.parquet")

print("PCA explained variance ratio: ", pca.explained_variance_ratio_.sum())

print("Fitting UMAP...")
reducer = UMAP(
    n_components=2,
    n_neighbors=UMAP_N_NEIGHBOURS,
    min_dist=UMAP_MIN_DIST,
    metric=METRIC,
    random_state=seed,
)
reducer.fit(pca.transform(features))
joblib.dump(reducer, "reducer.joblib")


reduce = lambda x: reducer.transform(pca.transform(x))

embedding = reduce(features).reshape((-1, 2))

df["umap_x"] = embedding[:, 0]
df["umap_y"] = embedding[:, 1]

print("Plotting figure...")

plt.rc("font", family="Liberation Mono")
alpha = 0.6
marker_size = 0.1
fig_size = (10 / 3, 3)
seed = 69
fontsize = 9


fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=df[["size"]].values,
    cmap="Spectral",
    marker=".",
    s=marker_size,
    vmin=25,
    vmax=70,
    alpha=alpha,
)

plt.gca().set_aspect("equal", "datalim")
# plt.axes(visible=False)
# plt.colorbar(boundaries=np.arange(0, 25) - 0.5).set_ticks(np.arange(0, 25))
cbar = fig.colorbar(scatter)
# cbar.set_label("source extension (arcsec)", rotation=270, size=25, labelpad=100)
cbar.ax.tick_params(labelsize=fontsize)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
fig.savefig("latent_space.png", bbox_inches="tight", pad_inches=0.08, dpi=600)


umap_cols = ["umap_x", "umap_y"]

# Save dataframe
print("Saving dataframe...")

df = df[["rgz_name", "umap_x", "umap_y", "ra", "dec", "size"]]

df = df.astype({col: "float32" for col in ["umap_x", "umap_y", "ra", "dec", "size"]})
df = df.astype({"rgz_name": "string"})


df.to_parquet("umap.parquet", index=False)
print("Done!")
