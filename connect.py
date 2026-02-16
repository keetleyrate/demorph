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
import itertools
import pickle
from network import load_network, plot_network, fuzzy_fractal, sholl
import pandas as pd
from utility import *
from analysis import * 


VIDEOPATH = "videos"
IMAGESPATH = "images"
RESULTS_PATH = "results"
C_EPS = 0.3
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
    thrsh = denoised > threshold_li(denoised)
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




class BranchDetector:

    def __init__(self, video, path, radius, lookahead, use_edges=False, ridge_low=5, ridge_high=50, ridge_blur=1, edge_low=25, edge_high=50, edge_sigma=1, blur_sigma=1, hat_size=3):
        self.path = path
        self.video = video
        self.frames = []
        self.detect_radius = radius
        self.lookahead = lookahead
        self.use_edges = use_edges
        self.ridge_low = ridge_low
        self.ridge_high = ridge_high
        self.ridge_blur = ridge_blur
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.edge_sigma = edge_sigma
        self.blur_sigma = blur_sigma
        self.hat_size = hat_size
        for fname in sorted(os.listdir(IMAGESPATH + "/" + path), key=lambda s: int(s.split("_")[1].split(".")[0])):
            raw = cv.imread(IMAGESPATH + "/" + path + "/" + fname, cv.IMREAD_GRAYSCALE)
            self.frames.append(raw)
        self.all_cells_mask, self.cell_masks, self.cell_centers, self.radi, self.L = get_cell_masks(self.frames[0])
        self.patch_kw = dict(
            patch_size=5,  # 5x5 patches
            patch_distance=6,  # 13x13 search area
        )


    def preproccess_frame(self, f):
        raw = self.frames[f]
        rescaled = exposure.rescale_intensity(raw, out_range=(0, 1))
        sigma = np.mean(restoration.estimate_sigma(rescaled))
        self.frames[f] = exposure.rescale_intensity(
            exposure.equalize_adapthist(
                restoration.denoise_nl_means(rescaled, h=0.6 * sigma, sigma=sigma, fast_mode=True, **self.patch_kw)
            ),
            out_range=(0, 255)
        )


    def write_preprocessed_frames(self):
        if not os.path.isdir(IMAGESPATH + "/" + self.path + "preprocessed"):
            os.mkdir(IMAGESPATH + "/" + self.path + "preprocessed")
        for f in tqdm(range(len(self.frames))):
            self.preproccess_frame(f)
            plt.imsave(IMAGESPATH + "/" + self.path + f"preprocessed/ppframe_{f}.png", self.frames[f])

    def read_preprocessed_frames(self):
        for f in tqdm(range(len(self.frames))):
            self.frames[f] = cv.imread(IMAGESPATH + "/" + self.path + f"preprocessed/ppframe_{f}.png", cv.IMREAD_GRAYSCALE)


    def set_edge_detection_parameters(self, frame):
        window_name = "Edges"
        edge_low_name = "Edge Low"
        edge_high_name = "Edge High"
        sigma_name = "Sigma"
        sigmas = np.linspace(0, 5, 128)
        cv.namedWindow(window_name)
        cv.createTrackbar(edge_low_name, window_name, 0, 255, nothing)
        cv.createTrackbar(edge_high_name, window_name, 1, 255, nothing)
        cv.createTrackbar(sigma_name, window_name, 0, 128, nothing)
        last_params = (None, None, None)
        to_show = np.copy(self.frames[frame])
        while True:
            key = cv.waitKey(1)
            params = tuple(cv.getTrackbarPos(name, window_name) for name in [edge_low_name, edge_high_name, sigma_name])
            if params != last_params:
                low, high, pos = params
                image = np.copy(self.frames[frame])
                edges = feature.canny(image, sigma=sigmas[pos], low_threshold=low, high_threshold=high)
                set_where(image, edges > 0, 255)
                to_show = image
                last_params = params
            cv.imshow(window_name, np.uint8(to_show))
            if key == 113:
                self.edge_low = low
                self.edge_high = low
                self.edge_sigma = sigmas[pos]
                print("Edge low:", low)
                print("Edge high:", high)
                print(f"Sigma: {sigmas[pos]:.4f}")
                break


    def set_edge_join_parameters(self, frame):
        window_name = "Blurring / Joining"
        sigma_name = "Sigma"
        top_hat_size = "TH Size"
        sigmas = np.linspace(0, 5, 128)
        cv.namedWindow(window_name)
        cv.createTrackbar(top_hat_size, window_name, 0, 64, nothing)
        cv.createTrackbar(sigma_name, window_name, 0, 128, nothing)
        last_params = (None, None)
        to_show = np.copy(self.frames[frame])
        edges = feature.canny(self.frames[frame], sigma=self.edge_sigma, low_threshold=self.edge_low, high_threshold=self.edge_high)
        while True:
            
            key = cv.waitKey(1)
            params = tuple(cv.getTrackbarPos(name, window_name) for name in [sigma_name, top_hat_size])
            if params != last_params:
                pos, hat_size = params
                image = np.copy(self.frames[frame])
                blobs = np.uint8(filters.gaussian(edges, sigmas[pos]) > 0)
                blobs = blobs - morphology.white_tophat(blobs, footprint=morphology.disk(hat_size))
                draw_outlines(image, blobs * 255, 255)
                to_show = image
                last_params = params
            cv.imshow(window_name, np.uint8(image))
            if key == 113:
                self.blur_sigma = sigmas[pos]
                self.hat_size = hat_size
                print(f"Blur Sigma: {sigmas[pos]:.4f}")
                print("Hat Radius:", hat_size)
                break


    def set_ridge_detection_parameters(self, frame):
        window_name = "Ridges"
        median_filter_name = "Median size"
        thresh_low_name = "Low"
        thresh_high_name = "High"
        sigma_name = "Sigma"
        color_name = "B - W"
        sigmas = np.linspace(0, 5, 128)
        cv.namedWindow(window_name)
        cv.createTrackbar(median_filter_name, window_name, 0, 16, nothing)
        cv.createTrackbar(thresh_low_name, window_name, 0, 255, nothing)
        cv.createTrackbar(thresh_high_name, window_name, 1, 255, nothing)
        cv.createTrackbar(sigma_name, window_name, 0, 128, nothing)
        cv.createTrackbar(color_name, window_name, 0, 1, nothing)
        last_params = (None, None, None)
        to_show = np.copy(self.frames[frame])
        while True:
            key = cv.waitKey(1)
            params = tuple(cv.getTrackbarPos(name, window_name) for name in [median_filter_name, thresh_low_name, thresh_high_name, sigma_name, color_name])
            if params != last_params:
                median_size, low, high, pos, b_or_w = params
                image = np.copy(self.frames[frame])
                blur = filters.gaussian(image, sigmas[pos])
                ridges = filters.meijering(blur, sigmas=range(1, 5), black_ridges=b_or_w)
                ridges = exposure.rescale_intensity(ridges, out_range=(0, 255))
                thresh = filters.apply_hysteresis_threshold(ridges, low, high)
                draw_outlines(image, thresh, 255)
                to_show = image
                last_params = params
            cv.imshow(window_name, np.uint8(image))
            if key == 113:
                print("Medain raidus:", median_size)
                print("Ridge Thresholds:", low, high)
                print(f"Sigma: {sigmas[pos]:.4f}")
                break

    
        


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
    
    def recompute_cell_masks_centered(self, sub_frame, center):
        ci, cj = center
        all_cells_mask, cell_masks, centers, radi, _ = get_cell_masks(sub_frame)
        i = min(range(len(centers)), key=lambda i: math.dist((ci, cj), centers[i]))
        return all_cells_mask, cell_masks[i], centers, radi
            




    def find_grown_cells(self):
        print("Finding grown cells ... ")
        self.grown_cells = [c for c in tqdm(range(len(self.cell_masks))) if self.has_grown(c)]

    def display_cells(self):
        ax = plt.axes()
        cpy = np.copy(self.frames[0])
        for i, mask in enumerate(self.cell_masks):
            borders = feature.canny(mask)
            xi, yi = np.where(borders > 0)
            cpy[xi, yi] = 255
            ax.text(*self.cell_centers[i], str(i))
        ax.imshow(cpy)
        plt.show()



    def find_branches_via_egdes(self, image, mask):
        edges = feature.canny(image, self.edge_sigma, self.edge_low, self.edge_high, mask)
        edges = np.uint8(filters.gaussian(edges, self.blur_sigma) > 0)
        edges = edges - morphology.white_tophat(edges, footprint=morphology.disk(self.hat_size))
        return edges
                        
    
    def find_branches_via_ridges(self, image):
        blur = filters.gaussian(image, self.ridge_blur)
        ridges = filters.meijering(blur, sigmas=range(1, 5), black_ridges=True)
        ridges = exposure.rescale_intensity(ridges, out_range=(0, 255))
        branches = filters.apply_hysteresis_threshold(ridges, self.ridge_low, self.ridge_high)
        return largest_connected_region(branches)
    

    
    def write_branch_maps(self, cell):
        path = f"maps/v-{self.video}/cell{cell}"
        if not os.path.exists(path):
            os.makedirs(path)
        for fi in tqdm(range(len(self.frames))):
            sub_frame, cell_mask, detect_mask, center = self.get_raduis_around_cell(cell, fi, self.detect_radius)
            mask = get_blob_mask_and_cell_map(sub_frame, center)
            if self.use_edges:
                branches = self.find_branches_via_egdes(sub_frame, mask)
            else:
                branches = self.find_branches_via_ridges(sub_frame) * mask
            write_map(path + f"/frame{fi}.txt", branches)


    def write_networks(self, cell):
        map_path = f"maps/v-{self.video}/cell{cell}"
        path = RESULTS_PATH + "/" + self.path + f"/networks/cell{cell}"
        if not os.path.exists(path):
            os.makedirs(path)
        B = []
        print("Loading maps ...")
        for i in tqdm(range(len(os.listdir(map_path)))):
            B.append(read_map(map_path + f"/frame{i}.txt"))
        B = np.array(B)
        sure = np.zeros(B[0].shape)
        _, _, _, center = self.get_raduis_around_cell(cell, 0, self.detect_radius)
        cell_map = np.zeros(sure.shape)
        cv.circle(cell_map, center, int(self.radi[cell] * 2), 1, -1)
        for i in tqdm(range(len(B))):
            window = np.sum(B[i : i + self.lookahead], axis=0)
            sure = (sure + (window >= math.floor(self.lookahead * 0.8))) > 0
            skel = morphology.skeletonize(sure)
            pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(np.uint8(skel), size=20)
            S = morphology.skeletonize(morphology.binary_closing(pruned_skeleton, morphology.disk(2)))
            tagged = tag_skelton(S)
            tagged[*np.where(np.logical_and(cell_map == 1, np.logical_or(tagged == 1, tagged == 2)))] = 3
            G = network_from_skeleton(tagged, self.L)
            with open(path + f"/frame_{i}_network.pkl", "wb") as infile:
                pickle.dump(G, infile)


    def show_maps(self, cell):
        map_path = f"maps/v-{self.video}/cell{cell}"
        B = []
        print("Loading maps ...")
        for i in tqdm(range(len(os.listdir(map_path)))):
            b = read_map(map_path + f"/frame{i}.txt")
            print(b.shape)
            B.append(b)
        sure = np.zeros(B[0].shape)
        _, _, _, center = self.get_raduis_around_cell(cell, 0, self.detect_radius)
        for i in tqdm(range(len(B))):
            window = np.sum(B[i : i + self.lookahead], axis=0)
            sure = (sure + (window >= math.floor(self.lookahead * 0.8))) > 0
            plt.cla()
            plt.clf()
            #skel = morphology.skeletonize(sure)
            cell_map = np.zeros(sure.shape)
            cv.circle(cell_map, center, int(self.radi[cell] * 2), 1, -1)
            #pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(np.uint8(skel), size=20)
            #S = morphology.skeletonize(morphology.binary_closing(pruned_skeleton, morphology.disk(2)))
            plt.imshow(sure + 2 * cell_map)
            plt.pause(0.1)


        



    def display_skeleton(self, tags, cell_center, image):
        G = network_from_skeleton(tags, cell_center, L=self.L)
        plot_network(G, image)
        # rgb_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # for i, j in zip(*np.where(tags > 0)):
        #     if tags[(i, j)] == 1:
        #         rgb_image[i, j] = (255, 0, 0)
        #     if tags[(i, j)] == 2:.
        #         rgb_image[i, j] = (0, 255, 0)
        #     if tags[(i, j)] > 2:
        #         rgb_image[i, j] = (0, 0, 255)
        # return rgb_image
    

    def has_grown(self, cell):
        d = int(3 * self.radi[cell])
        image, all_cells_mask, cell_mask, c = self.get_raduis_around_cell(cell, len(self.frames) - 1, d)
        #all_cells_mask, cell_mask, centers, radi = self.recompute_cell_masks_centered(image, c)
        #c = min(centers, key=lambda ci: math.dist(ci, c))
        if self.use_edges:
            branches = self.find_branches_via_egdes(image, all_cells_mask)
            connected = largest_connected_region(morphology.binary_closing(branches + util.invert(cell_mask), morphology.disk(2)))
        else:
            branches = self.find_branches_via_ridges(image) * all_cells_mask
            connected = largest_connected_region(morphology.binary_closing(branches + util.invert(cell_mask), morphology.disk(2)))
        S = morphology.skeletonize(connected)
        tagged = tag_skelton(S)
        node_mask = np.zeros(image.shape)
        cv.circle(node_mask, c, math.ceil(1.5 * self.radi[cell]), 1, -1)
        tagged[*np.where(np.logical_and(node_mask == 1, np.logical_or(tagged == 1, tagged == 2)))] = 3
        G = network_from_skeleton(tagged, self.L)
        return G.number_of_edges() >= 20 and G.number_of_nodes() >= MIN_JUNCTIONS
            
    
    def get_growth_radius(self, cell):
        assert cell in self.grown_cells
        if len(self.grown_cells) == 1:
            r = self.frames[0].shape[0] // 2
        else:
            closest_grown_cell, dist = min(
                enumerate(math.dist(self.cell_centers[cell], self.cell_centers[c]) for c in self.grown_cells if c != cell),
                key=lambda p: p[1]
            )
            r = int(dist - self.radi[closest_grown_cell] * 2.5)
        return max(r, int(self.radi[cell] * 4))



class DetectionJob:

    def __init__(self, video, radius, lookahead, tracked_cells, use_edges=False, ridge_low=5, ridge_high=50, ridge_blur=1, edge_low=25, edge_high=50, edge_sigma=1, blur_sigma=1, hat_size=3):
        self.tracked_cells = tracked_cells
        self.video = video
        self.detector = BranchDetector(video, f"V-{video}", radius, lookahead, use_edges, ridge_low, ridge_high, ridge_blur, edge_low, edge_high, edge_sigma, blur_sigma, hat_size)
    
    def run(self):
        self.detector.read_preprocessed_frames()
        for cell in self.tracked_cells:
            self.detector.write_branch_maps(cell)
            self.detector.write_networks(cell)
            write_all_feature_files(self.video)
    
    def show(self):
        for cell in self.tracked_cells:
            self.detector.show_maps(cell)
          



# jobs = [
#     DetectionJob(video=0, radius=250, lookahead=3, tracked_cells=[31, 33, 10, 16, 5, 19, 21], use_edges=False, ridge_low=10, ridge_high=80, ridge_blur=0.8),
#     DetectionJob(video=1, radius=500, lookahead=3, tracked_cells=[2], use_edges=False, ridge_low=10, ridge_high=50, ridge_blur=2.3),
#     DetectionJob(video=2, radius=600, lookahead=3, tracked_cells=[5], use_edges=True, edge_low=25, edge_high=45, edge_sigma=1.4, blur_sigma=0.63, hat_size=4),
#     DetectionJob(video=3, radius=500, lookahead=5, tracked_cells=[8], use_edges=True, edge_low=5, edge_high=50, edge_sigma=1.18, blur_sigma=0.6, hat_size=5),
#     DetectionJob(video=4, radius=500, lookahead=10, tracked_cells=[5, 6], use_edges=False, ridge_low=5, ridge_high=225, ridge_blur=1),
#     DetectionJob(video=5, radius=500, lookahead=10, tracked_cells=[2], use_edges=False, ridge_low=5, ridge_high=120, ridge_blur=0.75),
#     DetectionJob(video=6, radius=300, lookahead=10, tracked_cells=[2, 5], use_edges=True, edge_low=3, edge_high=25, edge_sigma=1.2, blur_sigma=0.7874, hat_size=5),
#     DetectionJob(video=7, radius=400, lookahead=5, tracked_cells=[4, 6], use_edges=True, edge_low=5, edge_high=25, edge_sigma=1.33, blur_sigma=0.7, hat_size=5)
# ]
# for job in jobs:
#     job.run()

#job = DetectionJob(video=5, radius=500, lookahead=10, tracked_cells=[2], use_edges=False, ridge_low=5, ridge_high=120, ridge_blur=0.75)
#job.detector.write_preprocessed_frames()
#job.run()
#job.show()


# D = BranchDetector(6, )

import heapq
import copy

class LCFSRidgeConnector:

    def __init__(self, ridges, points, sources):
        self.grid = ridges
        self.points = set(points)
        self.connected = set()
        self.queue = []
        self.visited = set()
        self.count = 0
        self.network = set(sources)

    def next(self):
        while len(self.queue) > 0:
            path = heapq.heappop(self.queue)[-1]
            last_edge = path[-1]
            _, tail, _ = last_edge
            if tail not in self.visited:
                self.visited.add(tail)
                return path
        
    def add(self, path, weight):
        last_edge = path[-1]
        _, tail, _ = last_edge
        if tail not in self.visited:
            heapq.heappush(
                self.queue,
                (weight, self.count, path)
            )
            self.count += 1

    def connect(self):
        unvistied_points = self.points - self.network
        if len(unvistied_points) == 0:
            return None
        seed = unvistied_points.pop()
        self.visited = set()
        self.queue = []
        self.add([(None, seed, 0)], 0)
        while len(self.queue) > 0:
            path = self.next()
            if path is None: # Ran out of unseen neighbours.
                # At this point we have done a BFS of a connected "ridge connected component"
                # If there are any other points visited in this attempt we can remove them from the set of considered points
                self.points -= self.visited
                return -1
            head, tail, _ = path[-1]
            if tail in self.network:
                break
            
            for new in neighbours(tail, self.grid):
                if self.grid[*new] > 0:
                    edge_weight = self.grid[*tail] - self.grid[*new] # Favour an increase in ridge value
                    first = path[0][1]
                    self.add(
                        path + [(tail, new, edge_weight)],
                        sum(w for (_, _, w) in path)
                    )

        points_in_path = []
        
        for head, tail, _ in path:
            if head is not None:
                points_in_path.append(head)
            points_in_path.append(tail)

        if len(points_in_path) == 0:
            self.points.remove(seed)
        else:
            self.network.update(points_in_path)





def get_ridge_map(image, min_ridge_value, min_local_max_value):
    gray = skimage.color.rgb2gray(image[:, :, :3])
    # Threshold
    mask_blobs = get_blob_mask_and_cell_map(gray, 0)
    ridge = filters.meijering(gray, sigmas=range(1, 5), black_ridges=True) * mask_blobs
    ax = plt.axes()
    ridge = exposure.rescale_intensity(ridge, out_range=(0, 255))
    mask = filters.apply_hysteresis_threshold(ridge, 10, 30)
    ridge *= mask

    # A* from every max to candidate point, stop early when connected 

    coords = feature.peak_local_max(ridge, min_distance=8)
    ax.imshow(ridge)
    for (x, y) in coords:
        if ridge[x, y] > min_local_max_value:
            ax.plot(y, x, 'r.', markersize=3)

    src = coords[np.argmax(ridge[*c] for c in coords)]
    xi, yi = src
    ax.plot(yi, xi, marker="x", markersize="6")

    search = LCFSRidgeConnector(ridge, list(map(tuple, coords)), tuple(src))
    
    last_network = copy.deepcopy(search.network)
    while search.connect() is not None:
        pass
    while last_network != search.network:
        print(len(search.points - search.network))
        last_network = copy.deepcopy(search.network)
        while search.connect() is not None:
            pass
    network_map = np.zeros(ridge.shape)
    for i, j in search.network:
        #ax.plot(j, i, 'b.', markersize=0.5)
        network_map[i, j] = 1
    network_map = morphology.skeletonize(network_map)
    T = tag_skelton(network_map)
    plt.imshow(T)
    plt.show()


# images = [plt.imread("test.png") for i in range(-1, 2)]
# get_ridge_map(sum(images), 0.02, 0.05)

# images = [skimage.color.rgb2gray(image[:, :, :3]) for image in images]
# #blur = filters.gaussian(image, self.ridge_blur)
# ridges = [filters.meijering(image, sigmas=range(1, 5), black_ridges=True) for image in images]
# plt.imshow(sum(ridges))
# plt.show()