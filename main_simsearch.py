import io
import joblib
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from umap import UMAP
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pickle
import torch
from einops import rearrange

from query_engine import get_image
from model import ResNet, BasicBlock
from main_explorelatent import remove_duplicates


def main():
    # TODO use tabs to allow more galaxies
    # TODO mke galaxies clickable for id_str with on_click callback

    st.set_page_config(layout="wide", page_title="VLA FIRST (RGZ DR1)")

    st.title("Radio Galaxy Representation Similarity Search")
    st.subheader("by [Inigo Slijepcevic](inigoval.github.io)")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    pca = joblib.load("pca.joblib")
    model = get_model()

    # df = load_umap()
    df = load_pca()
    print(df.columns)
    tree = fit_tree(df)

    with col1:
        n_images = st.slider(
            label="Number of images to show", min_value=4, max_value=20, value=12, step=4
        )

        duplicate_threshold = st.slider(
            label="Threshold for removing duplicates (arcsec)",
            min_value=0,
            max_value=200,
            value=30,
            step=1,
        )

        ra = st.number_input(label="RA value", format="%.6f", value=139.14801, step=0.000001)
        dec = st.number_input(label="DEC value", format="%.6f", value=4.6812873, step=0.000001)

        img = get_image(ra, dec, low=0.8, tensor=True)

        if img is not None:
            fig, ax = plt.subplots()
            ax.imshow(img.squeeze().numpy(), cmap="hot")
            ax.axis("off")
            fig.tight_layout()
            plt.show()
            st.pyplot(fig)
        else:
            st.warning("No image available for this location or timed out")

    with col2:
        if img is not None:
            x = model(rearrange(img, "c h w -> 1 c h w")).squeeze().detach().numpy()
            x = pca.transform(x.reshape(1, -1)).reshape(1, -1)

            # galaxies = get_closest_galaxies(df, tree, x[0], x[1], max_neighbours=500)
            galaxies = get_closest_galaxies(df, tree, x, max_neighbours=500)

            galaxies = remove_duplicates(galaxies, n_images, threshold=duplicate_threshold)

            show_gallery_of_galaxies(galaxies[:n_images])

            csv = convert_df(galaxies)

            st.download_button(
                label="Download CSV of the 500 galaxies closest to your search",
                data=csv,
                file_name="closest_500_galaxies.csv",
                mime="text/csv",
            )

    st.markdown("---")

    tell_me_more()


@st.cache_data
def get_model():
    resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], n_c=1, downscale=True, features=512)
    # weights = requests.get(
    #     "https://www.dropbox.com/s/ztvo4atyoq7k0xj/encoder_byol_resnet18.pt?dl=0?raw=1"
    # ).content

    # resnet.load_state_dict(torch.load(io.BytesIO(weights)))

    resnet.load_state_dict(torch.load("encoder.pt"))
    resnet.eval()

    return resnet


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun

    return df[["rgz_name", "ra", "dec"]].to_csv(index=False).encode("utf-8")


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=100)
    buf.seek(0)
    return Image.open(buf)


def get_closest_galaxies(df, tree, features, max_neighbours=1):
    _, indices = tree.kneighbors(features)
    indices = indices[0]  # will be wrapped in extra dim
    return df.iloc[indices[:max_neighbours]]


def show_gallery_of_galaxies(df):
    imgs = []
    captions = []

    for _, row in df.iterrows():
        img = get_image(row["ra"], row["dec"])
        captions.append(f"RA: {row['ra']}, Dec: {row['dec']}")

        img = img - img.min()
        img = img / img.max()

        imgs.append(img)

    # set up the plot grid
    assert len(df) % 4 == 0

    fig, axes = plt.subplots(nrows=int(len(df) / 4), ncols=4, figsize=(8, 8), facecolor="#0B0B0B")

    # plot each image on its corresponding axis and add a title
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="hot")
        # ax.set_title(captions[i])
        ax.axis("off")

    st.pyplot(fig)


@st.cache_data
def load_pca():
    df = pd.read_parquet("pca.parquet")
    return df


def fit_tree(df):
    # X = df[["umap_x", "umap_y"]]
    X = df[[f"pca_{i}" for i in range(200)]]
    # will always return 100 neighbours, cut list when used
    nbrs = NearestNeighbors(n_neighbors=500, algorithm="ball_tree").fit(X)
    return nbrs


def tell_me_more():
    st.markdown(
        """
    This app allows the user to query the internal representation of a self-supervised model trained on VLA FIRST images to find semantically similar data points. Code for training this model can be found [here](github.com/inigoval/byol). The 
    model's overparametrized representation is first compressed from 512 to 200 dimensions with PCA, which preserves over 99\% of the variance. The 200-dimensional embedding is then queried using an image scraped from SkyView 
    by using the user's input RA and DEC values. The input image can be any image from the VLA FIRST survey, and the app handles any pre-processing - all you need to do is specify the correct RA and DEC for your cutout.
    
    The galaxies shown are those closest to your input data-point in the 200 dimensional PCAlatent space, with the closest galaxies appearing first.

    The galaxies are drawn from the VLA FIRST RGZ DR1 data-set. Only galaxies with angular extent above 25 arcsec are shown.

    Note that displaying a larger number of images will result in slower performance (bottlenecked by SkyView query speed) and a larger probability of timeout. If the app hangs, it may have timed out 
    while querying SkyView. Refresh the page to try again.
    """
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)

    main()
