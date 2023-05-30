import io
import logging
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from streamlit_toggle import toggle

from query_engine import get_image


def update_center_coords(df):
    # on first run, default like so
    x_center = np.mean([df["umap_x"].min(), df["umap_y"].max()])
    y_center = np.mean([df["umap_x"].min(), df["umap_y"].max()])

    if "x" not in st.session_state:
        st.session_state["x"] = x_center
    if "y" not in st.session_state:
        st.session_state["y"] = y_center


def main():
    # TODO use tabs to allow more galaxies
    # TODO mke galaxies clickable for id_str with on_click callback
    st.set_page_config(layout="wide", page_title="VLA FIRST (RGZ DR1)")

    st.title("Radio Galaxy Representation Explorer")
    st.subheader("by [Inigo Slijepcevic](inigoval.github.io)")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        # add these first, as we need to have selectbox already to choose which to load
        st.markdown("## Move Around")
        st.markdown("Click anywhere on the latent space to view radio galaxies in nearby latent space.")

    df = load_umap()
    update_center_coords(df)

    tree = fit_tree(df)

    with col1:
        with st.empty():
            # will read current state
            new_coords = show_latent_space_interface(df)
            if new_coords:
                # if we had a click, update the state
                st.session_state["x"] = new_coords[0]
                st.session_state["y"] = new_coords[1]
                # rerun
                show_latent_space_interface(df)

        st.markdown("Location: ({:.3f}, {:.3f})".format(st.session_state["x"], st.session_state["y"]))

        n_cols = st.slider(
            label="Number of columns of images", min_value=1, max_value=8, value=2, step=1
        )

        n_images = st.slider(
            label="Number of images to show",
            min_value=n_cols,
            max_value=n_cols * 10,
            value=n_cols,
            step=n_cols,
        )

        # filter_duplicates = toggle(label="Filter duplicates", key="duplicates", value=True)

        duplicate_threshold = st.slider(
            label="Threshold for removing duplicates (arcsec)",
            min_value=0,
            max_value=1000,
            value=30,
            step=1,
        )

    with col2:
        # read the state
        galaxies = get_closest_galaxies(
            df, tree, st.session_state["x"], st.session_state["y"], max_neighbours=500
        )

        # if remove_duplicates:
        galaxies = remove_duplicates(galaxies, n_images, threshold=duplicate_threshold)

        show_gallery_of_galaxies(galaxies[:n_images], n_cols)

        csv = convert_df(galaxies)

        st.download_button(
            label="Download CSV of the 500 galaxies closest to your search",
            data=csv,
            file_name="closest_500_galaxies.csv",
            mime="text/csv",
        )

    st.markdown("---")

    tell_me_more()


def tell_me_more():
    st.markdown(
        """
    This explorer visualises the internal representation of a self-supervised model trained on VLA FIRST images. Code for training this model can be found [here](github.com/inigoval/byol). The 
    model's overparametrized representation is first compressed from 512 to 200 dimensions with PCA, which preserves over 99\% of the variance. The 200-dimensional embedding is then visualised with UMAP.
    The galaxies shown are those closest to your selected point in latent space, with the closest galaxies appearing first.

    The galaxies are drawn from the VLA FIRST RGZ DR1 data-set. Only galaxies with angular extent above 25 arcsec are shown.

    Note that displaying a larger number of images will result in slower performance (bottlenecked by SkyView query speed) and a larger probability of timeout. If the app hangs, it may have timed out 
    while querying SkyView. Click on the latent space again to reset.
    """
    )


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun

    return df[["rgz_name", "ra", "dec"]].to_csv(index=False).encode("utf-8")


def show_latent_space_interface(df):
    fig, ax = plot_latent_space(df, st.session_state["x"], st.session_state["y"])
    fig.tight_layout()

    # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
    im = fig_to_pil(fig)

    # https://github.com/blackary/streamlit-image-coordinates
    x_y_dict = streamlit_image_coordinates(im)  # None or {x: x, y: y}

    if x_y_dict is None:  # no click yet, return None
        return None
    else:
        # was a click, return as coordinates
        x_pix, y_pix = x_y_dict["x"], x_y_dict["y"]
        x, y = ax.transData.inverted().transform([x_pix, y_pix])

        # Hacky solution to align mouse pointer with latent space image
        offset = 13.48

        y = -y + offset
        return x, y


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=100)
    buf.seek(0)
    return Image.open(buf)


def plot_latent_space(df, x=None, y=None, figsize=(5.6, 5.6), data_pad=0.1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df["umap_x"], df["umap_y"], s=0.55, alpha=0.1)
    if x is not None:
        ax.scatter(x, y, marker="+", c="r")
    plt.xlim(df["umap_x"].min() - data_pad, df["umap_x"].max() + data_pad)
    plt.ylim(df["umap_y"].min() - data_pad, df["umap_y"].max() + data_pad)
    plt.axis("off")
    return fig, ax


def get_closest_galaxies(df, tree, x, y, max_neighbours=1):
    # st.markdown(x)
    # st.markdown(y)
    _, indices = tree.kneighbors(np.array([x, y]).reshape(1, 2))
    indices = indices[0]  # will be wrapped in extra dim
    return df.iloc[indices[:max_neighbours]]


def remove_duplicates(df, n, threshold=0.1):
    i = 1
    while i < n:
        ra, dec = df.iloc[:i]["ra"].values, df.iloc[:i]["dec"].values
        diff_ra = np.abs(ra - df.iloc[i]["ra"]) * np.cos(dec)
        diff_dec = np.abs(dec - df.iloc[i]["dec"])

        diff = ((diff_ra**2 + diff_dec**2) ** 0.5) < threshold / 3600

        if np.any(diff):
            idx_dup = np.argwhere(diff)
            print(f"{len(idx_dup)} duplicates found")
            print(f"Removing duplicate {df.iloc[i]['rgz_name']} and {df.iloc[idx_dup[0]]['rgz_name']}")
            df = df.drop(df.index[i])
        else:
            i += 1

    return df


def show_gallery_of_galaxies(df, n_cols):
    imgs = []
    captions = []

    for _, row in df.iterrows():
        img = get_image(row["ra"], row["dec"])
        captions.append(f"RA: {row['ra']}, Dec: {row['dec']}")

        img = img - img.min()
        img = img / img.max()

        imgs.append(img)

    # st.image(imgs, width=200)

    # set up the plot grid
    # assert len(df) % 4 == 0

    fig, axes = plt.subplots(
        nrows=int(len(df) / n_cols),
        ncols=n_cols,
        figsize=(8, 8),
        facecolor="#0B0B0B",
    )

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # plot each image on its corresponding axis and add a title
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="hot")
        # ax.set_title(captions[i])
        ax.axis("off")

    st.pyplot(fig)


@st.cache_data
def load_umap():
    df = pd.read_parquet("umap.parquet")
    return df


def fit_tree(df):
    X = df[["umap_x", "umap_y"]]
    # will always return 100 neighbours, cut list when used
    nbrs = NearestNeighbors(n_neighbors=500, algorithm="ball_tree").fit(X)
    return nbrs


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)

    main()
