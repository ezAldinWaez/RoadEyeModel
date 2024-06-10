import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib widget
import cv2 as cv
import os

IMG_DIR = './img'
OUT_DIR = './out'
DS_DIR = '../dataset'

os.makedirs(f'{OUT_DIR}', exist_ok=True)


def destroy_when_esc():
    while True:
        if cv.waitKey(0) & 0xFF == 27:
            return cv.destroyAllWindows()


def img_to_canny_edges(img, blur_kernel=None):
    """
    Detect the edges of the given `img` using Canny algorithm,
    the upper and lower values are calculated depending on img
    median value.

    You can bluring the image before applying Canny by parsing
    a `blur_kernel` to it (e.g. blur_kernel=(5, 5)).

    You can change `blur_kernel` size to control how much
    detailes are detected.
    """
    if blur_kernel:
        img = cv.blur(img, blur_kernel)
    img_med = np.median(img)
    img_med_lower = int(max(0, 0.7 * img_med))
    img_med_upper = int(min(255, 1.3 * img_med))
    img_edges = cv.Canny(img, img_med_lower, img_med_upper)
    return img_edges


def edges_to_contours(edges, rng_down=0, rng_up=np.inf, color=(255, 0, 0), thickness=1):
    """
    Find and draw contours from a given `edges`, the contours
    are drawen on black image with the given `color` and
    `thickness`.

    You can change `rng_down` and `rng_up` to determain the
    range that contours' lengths are belong to.
    """
    all_contours = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    selected_contours = [c for c in all_contours if rng_down < len(c) < rng_up]
    img_contours = np.zeros((*edges.shape[:2], 3)).astype('uint8')
    cv.drawContours(img_contours, selected_contours, -1, color, thickness)
    return img_contours


def img_to_dist_gray(img, base_color):
    """
    Convert the given `img` to grayscale depending on how
    much each pixel is difference with the `base_color`.
    """
    hight, width, num_channels = img.shape
    pixels = img.reshape((hight * width, num_channels))
    img_dist_gray = np.array([
        np.sqrt(np.sum((base_color - pixel) ** 2))
        for pixel in pixels
    ]).reshape(hight, width)
    img_dist_gray_norm = cv.normalize(
        src=img_dist_gray, dst=None, alpha=0, beta=255,
        norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U,
    )
    return img_dist_gray_norm


def find_shadow_mask(img_rgb, cln_kernel=np.ones((3, 3), np.uint8)):
    """
    ...
    """
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)

    mean_l = np.mean(img_lab[:, :, 0])
    mean_a = np.mean(img_lab[:, :, 1])
    mean_b = np.mean(img_lab[:, :, 2])
    std_l = np.std(img_lab[:, :, 0])

    shadow_mask = (
        (img_lab[:, :, 0] <= (mean_l - std_l / 3))
        if mean_a + mean_b <= 256 else
        ((img_lab[:, :, 0] < mean_l) & (img_lab[:, :, 2] < mean_b))
    ).astype(np.uint8) * 255

    shadow_mask_cln = cv.morphologyEx(shadow_mask, cv.MORPH_OPEN, cln_kernel)
    shadow_mask_cln = cv.dilate(shadow_mask_cln, cln_kernel, iterations=1)
    shadow_mask_cln = cv.erode(shadow_mask_cln, cln_kernel, iterations=1)

    return shadow_mask_cln


def shadow_remove(img, shadow_mask):
    """
    ...
    """
    img_float = img.astype(np.float32)

    in_shadow_pixels = np.where(shadow_mask)
    out_shadow_pixels = np.where(~shadow_mask)

    avg_rgb_in_shadow = np.array([
        np.mean(img_float[*in_shadow_pixels, c])
        for c in range(3)
    ])
    avg_rgb_out_shadow = np.array([
        np.mean(img_float[*out_shadow_pixels, c])
        for c in range(3)
    ])
    constants = avg_rgb_out_shadow / avg_rgb_in_shadow

    for c in range(3):
        img_float[*in_shadow_pixels, c] *= constants[c]

    img_shadow_removed = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_shadow_removed


def thresh_bin(img, thresh=None):
    if thresh is None:
        thresh = np.mean(img)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.threshold(img_gray, thresh, 255, cv.THRESH_BINARY)[1]
    return img_bin.astype('uint8')


def thresh_bin_inv(img, thresh=None):
    if thresh is None:
        thresh = np.mean(img)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.threshold(img_gray, thresh, 255, cv.THRESH_BINARY_INV)[1]
    return img_bin.astype('uint8')


def thresh_otsu(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.threshold(
        img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    return img_bin.astype('uint8')


def thresh_otsu_inv(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.threshold(
        img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    return img_bin.astype('uint8')


def thresh_adapt_mean(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.adaptiveThreshold(
        img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return img_bin.astype('uint8')


def thresh_adapt_gaussian(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if len(
        img.shape) > 2 else img
    img_bin = cv.adaptiveThreshold(
        img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img_bin.astype('uint8')


def img_to_mexican_hat(img, ksize=3):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = cv.magnitude(sobel_x, sobel_y)
    return sobel_combined


def img_to_hanny(img, blur_kernal=(3, 3)):
    img_blur = cv.GaussianBlur(img, blur_kernal, 0)
    img_blur_lap = cv.Laplacian(img_blur, cv.CV_64F)
    return (img_blur_lap * 255).astype('uint8')

# * Others


def scale_img(img, scale=0.5):
    new_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img_scaled = cv.resize(img, new_shape)
    return img_scaled


def get_video_background(vedio_path):
    frames = [plt.imread(f'{vedio_path}/{file}')
              for file in os.listdir(vedio_path) if file.endswith('.jpg')]
    background = np.median(frames, axis=0).astype('uint8')
    return background


def img_to_pixels(img):
    return img.flatten().reshape(1, -1)


def evaluate(img_result, img_desired):
    l1, l2 = img_result.flatten(), img_desired.flatten()

    return {
        'Correlation Coeficient': f'{np.corrcoef(l1, l2)[0, -1] :.3%}',
        'Eculodian Distance': f'{np.sqrt(np.sum((l1 - l2) ** 2)) :,.6}',
        'Cosine Similarity': f'{np.sum(l1 * l2, axis=-1) / (np.linalg.norm(l1, axis=-1) * np.linalg.norm(l2, axis=-1)) :f}',
    }