import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_li
from skimage import filters, restoration, morphology, feature, util, exposure
import scipy.sparse.csgraph as graph
from tqdm import tqdm
import math
import copy
import itertools
import pickle
from network import load_network, plot_network, fuzzy_fractal, sholl
import pandas as pd
from plantcv import plantcv as pcv
from utility import *
from analysis import *
from connect import LCFSRidgeConnector
from network import graph_from_skeleton, plot_graph
from PIL import Image
import multiprocessing
import time

VIDEOPATH = "videos"
IMAGESPATH = "images"
RESULTS_PATH = "results"
C_EPS = 0.5
MIN_CELL_AREA = 100
BRANCH_CHECK_OFFSET = 10
BRANCH_CHECK_RADIUS = 50
MIN_CELL_SIZE = 50
MIN_BRANCH_LENGTH = 16
MIN_GROWN_BRANCH_LENGTH = 16
MIN_JUNCTIONS = 16
MIN_NODE_DIST = 20
MIN_CELL_RAD = 8
MIN_CENTER_DIST = 100

index = 0


def rename_frames(path):
    for i, fname in enumerate(sorted(os.listdir(path))):
        os.rename(path + "/" + fname, path + f"/frame_{i}.tif" )

def preproccess_frame(args):
    path, raw, f, patch_kw = args
    rescaled = exposure.rescale_intensity(raw, out_range=(0, 1))
    sigma = np.mean(restoration.estimate_sigma(rescaled))
    prep = exposure.rescale_intensity(
        exposure.equalize_adapthist(
            restoration.denoise_nl_means(rescaled, h=0.6 * sigma, sigma=sigma, fast_mode=True, **patch_kw)
        ),
        out_range=(0, 255)
    )
    plt.imsave(path + f"preprocessed/ppframe_{f}.png", prep)


def write_frames_from_videos():
    video_num = 0
    for vid in os.listdir(VIDEOPATH):
        capture = cv.VideoCapture()
        capture.open(VIDEOPATH + "/" + vid)
        img_num = 0
        while (t := capture.read())[0]:
            _, img = t
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            print(img)
            if not os.path.isdir(IMAGESPATH + "/" + f"V-{video_num}"):
                os.mkdir(IMAGESPATH + "/" + f"V-{video_num}")
            cv.imwrite(IMAGESPATH + "/" + f"V-{video_num}" + "/" + f"frame_{img_num}.png", img)
            img_num += 1
        video_num += 1

def get_contours(image):
    denoised = cv.GaussianBlur(image, ksize=(3, 3), sigmaX=4)
    thrsh = denoised > 150#threshold_li(denoised)
    #thrsh = morphology.binary_erosion(thrsh, morphology.disk(4))
    contours, _ = cv.findContours(np.uint8(thrsh), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def get_blob_mask_and_cell_map(image, orgin):
    contours = get_contours(image)
    mask = np.ones(image.shape)
    cv.drawContours(mask, contours, -1, 0, -1)
    #mask = np.uint8(morphology.binary_erosion(mask, morphology.disk(16)))
    return mask


def get_cell_masks(image):
    contours = get_contours(image)
    cell_masks = []
    centers = []
    radi = []
    blob_masks = []
    cont_centers = []
    plt.imsave("inital.png", image)
    cnts = np.zeros(image.shape)
    cv.drawContours(cnts, contours, -1, 255, -1)
    plt.imsave("contours.png", cnts)

    for c in contours:
        center, radius = cv.minEnclosingCircle(c)
        center = tuple(map(int, center))
        cont_centers.append(center)
        zero_in_cell_mask = np.ones((image.shape))
        cv.drawContours(zero_in_cell_mask, [c], 0, 0, -1)
        inner_zeros = np.ones(image.shape)
        cv.circle(inner_zeros, center, int(radius), 0, -1)
        blob_masks.append(inner_zeros)
        if radius >= MIN_CELL_RAD and cirularity(c) >= C_EPS:
            radi.append(radius)
            centers.append(center)
            cell_masks.append(inner_zeros)
    r = np.array(radi)
    weighted_mean = np.dot(r, r) / np.sum(r)
    to_keep = r >= weighted_mean - 2 * np.std(radi)
    cell_masks = [m for i, m in enumerate(cell_masks) if to_keep[i]]
    centers = [c for i, c in enumerate(centers) if to_keep[i]]
    radi = [r for i, r in enumerate(radi) if to_keep[i]]
    detection_mask = math.prod(m for m in blob_masks)
    L = min(math.dist(c1, c2) for c1, c2 in itertools.combinations(cont_centers, r=2))
    return detection_mask, cell_masks, centers, radi, L



def compute_tree_features(tree):
    num_branches = 0
    num_junctions = 0
    lengths = []
    max_dist = 0
    center = tree.shape[0] // 2, tree.shape[1] // 2
    labels, num = skimage.measure.label(tree, connectivity=2, return_num=True)
    for tx, ty in [np.where(labels == i) for i in range(1, num + 1)]:
        tag = tree[tx[0], ty[0]]
        if tag == 2 and len(tx) > MIN_BRANCH_LENGTH:
            num_branches += 1
            lengths.append(len(tx))
        if tag > 2:
            num_junctions += 1
    return num_branches, num_junctions, lengths, 0

def load_ridge_map(path, cell, frame):
    return np.asarray(Image.open(f"cache/{path}/cell{cell}/frame{frame}_ridge.png").convert('L')).astype(float) 

def write_ridge_map(args):
    image, path, cell, frame, black_ridges = args
    plt.imsave(f"cache/{path}/cell{cell}/frame{frame}.png", image)
    mask_blobs = get_blob_mask_and_cell_map(image, 0)
    ridge = filters.meijering(image, sigmas=range(1, 5), black_ridges=black_ridges) * mask_blobs
    ridge = exposure.rescale_intensity(ridge, out_range=(0, 255))
    plt.imsave(f"cache/{path}/cell{cell}/frame{frame}_ridge.png", ridge)

def skeletonize_via_ridge_following(args):
    num_frames, ridge_low, ridge_high, window_size, alpha, path, min_max_dist, rel_max, prop_src, src_center, cell, frame = args
    N = num_frames - 1
    offset = math.floor(window_size / 2)
    left = max(0, frame - offset)
    right = min(N, frame + offset)
    window = np.array(range(left - frame, right - frame + 1))
    weights = np.exp(-alpha * window**2)
    weights /= np.sum(weights)
    maps = [load_ridge_map(path, cell, frame + w) for w in window]
    ridge = np.sum(maps * weights[:, np.newaxis, np.newaxis], axis=0)
    ridge = exposure.rescale_intensity(ridge, out_range=(0, 255))
    mask = filters.apply_hysteresis_threshold(ridge, ridge_low, ridge_high)
    ridge *= mask
    coords = feature.peak_local_max(ridge, min_distance=min_max_dist, threshold_rel=rel_max)
    srcs = sorted(coords, key=lambda c: math.dist(src_center, c))[:math.ceil(len(coords) * prop_src)]



    search = LCFSRidgeConnector(ridge, list(map(tuple, coords)), list(map(tuple, srcs)))
    last_network = copy.deepcopy(search.network)
    while search.connect() is not None:
        pass
    while last_network != search.network:
        last_network = copy.deepcopy(search.network)
        while search.connect() is not None:
            pass
    smap = np.zeros(ridge.shape)
    for p in search.network:
        smap[*p] = 1
    smap = morphology.skeletonize(smap)
    skeleton = np.array(np.where(smap > 0))
    image = plt.imread(f"cache/{path}/cell{cell}/frame{frame}.png")
    spath = f"cache/{path}/cell{cell}"
    if not os.path.exists(spath):
        os.makedirs(spath)
    np.savetxt(f"cache/{path}/cell{cell}/frame{frame}.txt", list(zip(*skeleton)))


   



class BranchDetector:

    def __init__(self, path, black_ridges=True, num_cpus=1):
        self.path = path
        self.num_cpus = num_cpus
        self.frames = []
        self.ridge_low = 10
        self.ridge_high = 30
        self.min_max_dist = 5
        self.rel_max = 0.2
        self.prop_src = 0.03
        self.black_ridges = black_ridges
        self.window_size = 5
        self.alpha = 0.1
        self.tracked_cells = []
        
        for fname in sorted(os.listdir(path), key=lambda s: int(s.split("_")[1].split(".")[0])):
            raw = cv.imread(path + "/" + fname, cv.IMREAD_GRAYSCALE)
            self.frames.append(raw)
        self.all_cells_mask, self.cell_masks, self.cell_centers, self.radi, self.L = get_cell_masks(self.frames[0])
        self.patch_kw = dict(
            patch_size=5,  # 5x5 patches
            patch_distance=6,  # 13x13 search area
        )




    def write_preprocessed_frames(self):
        if not os.path.isdir(self.path + "preprocessed"):
            os.mkdir(self.path + "preprocessed")
        num_frames = len(self.frames)
        args_generator = ((self.path, self.frames[f], f, self.patch_kw) for f in range(num_frames))
        print(f"Preprocessing images on {self.num_cpus} Cores ...")
        with multiprocessing.Pool(processes=self.num_cpus) as pool:
            for _ in tqdm(pool.imap_unordered(preproccess_frame, args_generator), total=num_frames):
                pass



    def read_preprocessed_frames(self):
        for f in tqdm(range(len(self.frames))):
            self.frames[f] = cv.imread(self.path + f"preprocessed/ppframe_{f}.png", cv.IMREAD_GRAYSCALE)


    def get_raduis_around_cell(self, cell, frame, raduis):
        n, m = self.frames[frame].shape
        ci, cj = self.cell_centers[cell]
        left, right = max(0, ci - raduis), min(m, ci + raduis + 1)
        top, bottom = max(0, cj - raduis), min(n, cj + raduis + 1)
        sub_frame = self.frames[frame][top : bottom, left : right]
        # Recompute detection mask, refine the cell center
        return (
            sub_frame,
            self.cell_masks[cell][top : bottom, left : right], 
            self.all_cells_mask[top : bottom, left : right],
            (ci - left, cj - top)
        )


    def write_ridge_maps(self, cell, detect_radius):
        if not os.path.exists(f"cache/{self.path}/cell{cell}"):
            os.makedirs(f"cache/{self.path}/cell{cell}", exist_ok=True)
        num_frames = len(self.frames)
        args_generator = (
            (
                self.get_raduis_around_cell(cell, f, detect_radius)[0], 
                self.path, 
                cell, 
                f, 
                self.black_ridges
            )
            for f in range(num_frames)
        )
        print(f"Writing ridge maps on {self.num_cpus} cores ...")
        with multiprocessing.Pool(processes=self.num_cpus) as pool:
            for _ in tqdm(pool.imap_unordered(write_ridge_map, args_generator), total=num_frames):
                pass
    




    def load_ridge_maps(self, cell):
        self.rmaps = [
            np.asarray(
                Image.open(f"cache/{self.path}/cell{cell}/frame{f}_ridge.png")
                .convert('L'))
                .astype(float) 
                for f in range(len(self.frames)
            )
        ]

    def get_source_center(self, cell):
        n, m, _ = self.frames[0].shape
        ci, cj = self.cell_centers[cell]
        radius = next(r for c, r in self.tracked_cells if c == cell)
        left, right = max(0, ci - radius), min(m, ci + radius + 1)
        top, bottom = max(0, cj - radius), min(n, cj + radius + 1)
        return int(cj - top), int(ci - left)

    
    def write_skeletons(self, cell, frames=[]):
        self.load_ridge_maps(cell)
        center = next(src for c, _, src in self.tracked_cells if c == cell)
        args_generator = (
            (
                len(self.frames),
                self.ridge_low,
                self.ridge_high,
                self.window_size,
                self.alpha,
                self.path,
                self.min_max_dist,
                self.rel_max,
                self.prop_src,
                center,
                cell,
                frame
            ) for frame in (frames if len(frames) > 0 else range(len(self.frames)))
        )
        print(f"Writing skeletons on {self.num_cpus} cores ...")  
        with multiprocessing.Pool(processes=self.num_cpus) as pool:
            for _ in tqdm(pool.imap_unordered(skeletonize_via_ridge_following, args_generator), total=len(self.frames)):
                pass



    def write_all_networks(self, cell):
        network_path = "results/" + self.path + f"/cell{cell}/networks"
        if not os.path.exists(network_path):
            os.makedirs(network_path, exist_ok=True)
        args_generator = (
            (
                self.path,
                cell,
                frame
            ) for frame in range(len(self.frames))
        )
        print(f"Writing graphs on {self.num_cpus} cores ...")
        with multiprocessing.Pool(processes=self.num_cpus) as pool:
            for _ in tqdm(pool.imap_unordered(graph_from_skeleton, args_generator), total=len(self.frames)):
                pass


class DetectionJob:

    def __init__(self, path, black_ridges=True, num_cpus=1):
        self.detector = BranchDetector(path, black_ridges, num_cpus)

    def run(self, graphs_only=False):
        for cell, r, _ in self.detector.tracked_cells:
            if not graphs_only:
                self.detector.write_skeletons(cell)
            self.detector.write_all_networks(cell)