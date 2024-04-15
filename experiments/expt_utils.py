import numpy as np
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
