import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import os

IMG_DIR = './img'
OUT_DIR = './out'
DS_DIR = '../dataset'


def make_out_dir(exptNum: str) -> str:
    OUT_DIR = f"out/{exptNum}"
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def destroy_when_esc():
    while (cv.waitKey(0) & 0xFF) != 27:
        continue
    cv.destroyAllWindows()


# * Usful Functions from Experiment 01 *


def img_to_canny_edges(img, blur_kernel=(5, 5)):
    img_med = np.median(img)
    img_med_lower = int(max(0, 0.7 * img_med))
    img_med_upper = int(min(255, 1.3 * img_med))
    img_blur = cv.blur(img, blur_kernel)
    img_edges = cv.Canny(img_blur, img_med_lower, img_med_upper)
    return img_edges


def edges_to_contours(edges, color=(0, 0, 255), thickness=2):
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros((*edges.shape[:2], 3))
    cv.drawContours(img_contours, contours, -1, color, thickness)
    return img_contours


# * Usful Functions from Experiment 07 *


def img_to_mexican_hat(img, ksize=3):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = cv.magnitude(sobel_x, sobel_y)
    return sobel_combined


def img_to_hanny(img, blur_kernal=(3, 3)):
    img_blur = cv.GaussianBlur(img, blur_kernal, 0)
    img_blur_lap = cv.Laplacian(img_blur, cv.CV_64F)
    return img_blur_lap

def imge_otsu_thresh(img):
    img_gray = img if (len(img.shape) < 3) else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_gray_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img_gray_thresh