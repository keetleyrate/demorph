"""
File: utility.py
Author: Keetley Rate
Role: Research Intern OIST
Date: February 16, 2026
Description:
    Image processing utilities for cell segmentation and skeleton 
    analysis. Provides functions for contour quantification, blob filtering, 
    and I/O for coordinate-based spatial maps.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage
from PIL import Image
from skimage import measure, feature


def show_image(img):
    img = plt.imshow(img)
    plt.colorbar(img)
    plt.show()

def draw_outlines(image, binary, color):
    outline = feature.canny(binary) > 0
    set_where(image, outline > 0, 255 if color > 0 else 0)

def set_where(a, cond, value):
    a[*np.where(cond)] = value

def cirularity(c):
    return 4 * np.pi * cv.contourArea(c) / cv.arcLength(c, True) ** 2

def remove_small_blobs(blobs, T):
    labels, num = measure.label(blobs, connectivity=2, return_num=True)
    for i in range(1, num + 1):
        xi, yi = np.where(labels == i)
        if len(xi) < T:
            blobs[xi, yi] = 0


def neighbours(p, image):
    n, m = image.shape
    i, j = p
    cross = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] 
    diag = [(i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
    cands = cross + diag
    return filter(lambda p: 0 <= p[0] <= n - 1 and 0 <= p[1] <= m - 1, cands)

def neighbours_no_image(p):
    i, j = p
    cross = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] 
    diag = [(i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
    return cross + diag


def tag_skelton(skelton):
    tags = {}
    if len(skelton.shape) == 1:
        skelton = np.array([skelton])
    skelton = set(map(tuple, skelton))
    for p in skelton:
        tags[p] = min(sum(int(n in skelton) for n in neighbours_no_image(p)), 3)
    return tags


def nothing(x):
    pass

def largest_connected_region(binary_image):
    out = np.zeros(binary_image.shape)
    labels, num = skimage.measure.label(binary_image, connectivity=2, return_num=True)
    postions = [np.where(labels == i) for i in range(1, num + 1)]
    if len(postions) > 0:
        x, y = max(postions, key=lambda p: len(p[0]))
        out[x, y] = 1
    return out

def write_map(filename, M):
    n, m = M.shape
    x, y = np.where(M > 0)
    np.savetxt(filename, np.column_stack((x, y)), header=f"{n},{m}", comments="", fmt="%i")

def read_map(filename):
    n, m = np.loadtxt(filename, max_rows=1, delimiter=",", dtype=int)
    A = np.zeros((n, m))
    d = np.loadtxt(filename, skiprows=1, dtype=int, delimiter=" ")
    if len(d.shape) != 2:
        A[*d] = 1
    else:
        A[d[:, 0], d[:, 1]] = 1
    return A

def load_image_as_float(path):
    return np.asarray(Image.open(path).convert('L')).astype(float)