from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u


import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T


def rescale_image(img, low):
    img_max = np.max(img)
    img_min = low * 1e-3
    # img -= img_min
    img /= max(1e-6, img_max - img_min)  # clamp divisor so it can't be zero
    # img /= img_max - img_min
    img *= 255.0

    return img


def crop_centre(img, crop=150):
    xsize = np.shape(img)[0]  # image width
    ysize = np.shape(img)[1]  # image height
    startx = xsize // 2 - (crop // 2)
    starty = ysize // 2 - (crop // 2)
    sub_img = img[startx : startx + crop, starty : starty + crop]

    return sub_img


def apply_circular_mask(img, maj, frac=0.6):
    centre = (np.rint(img.shape[0] / 2), np.rint(img.shape[1] / 2))
    maj = frac * maj / 1.8  # arcsec --> pixels

    Y, X = np.ogrid[: img.shape[1], : img.shape[1]]
    dist_from_centre = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    mask = dist_from_centre <= maj

    img *= mask.astype(int)

    return img


def get_image(ra: float, dec: float, low: float = 0, tensor=False):
    # Get fits file from ra and dec
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    # url = SkyView.get_image_list(position=sky, survey=["VLA FIRST (1.4 GHz)"], cache=False)
    # try:
    # hdu = requests.get(url[0], allow_redirects=True).content
    # except:
    # logging.info("No FITS available")
    # return None
    try:
        hdu = SkyView.get_images(position=sky, survey=["VLA FIRST (1.4 GHz)"])[0][0]
    except:
        print("Failed to query SkyView, most likely timeout.")

    # Open fits file
    # print(fitsfile)
    # hdu = fits.open(fitsfile)
    # try: hdu = fits.open(fitsfile)
    # except:
    # logging.info("Corrupt FITS file")
    # return None

    data = np.squeeze(hdu.data)
    # pl.subplot(161)
    # pl.imshow(data)
    # pl.title("Orig")

    # crop centre:
    img = crop_centre(data, crop=150)
    # pl.subplot(162)
    # pl.imshow(img)
    # pl.title("Crop")

    # radial crop:
    # img = apply_circular_mask(img, maj)
    # pl.subplot(163)
    # pl.imshow(img)
    # pl.title("Mask")

    # Make writeable
    img = img.copy().astype(np.float32)

    # remove nans:
    img[np.where(np.isnan(img))] = 0.0
    # pl.subplot(164)
    # pl.imshow(img)
    # pl.title("NaN")

    # subtract 3 sigma noise:
    img[np.where(img <= low * 1e-3)] = 0.0
    # pl.subplot(165)
    # pl.imshow(img)
    # pl.title("Sigma clip")

    # rescale image:
    img = rescale_image(img, low)
    # pl.subplot(166)
    # pl.imshow(img)
    # pl.title("Rescale")
    # pl.show()

    img = array_to_png(img)

    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((0.008008896,), (0.05303395,)),
        ]
    )

    img = transform(img)

    if not tensor:
        img = img.squeeze().numpy()

    return img


def array_to_png(img):
    im = Image.fromarray(img)
    im = im.convert("L")

    return im
