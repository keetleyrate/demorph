"""
File: demorph.py
Author: Keetley Rate
Role: Research Intern OIST
Email: keetleyjames@gmail.com
Github: https://github.com/keetleyrate
Date: February 16, 2026
Description: 
    GUI tool for dendrite morphology analysis. This script 
    allows for manual cell selection, ridge detection parameter tuning,
    and source point placement. The script outputs graphs representing
    dendrite morphology as networkx Graph() objectsas serialised with
    pickle.

Usage:
    python demorph_gui.py [path] --rcol [b/w] [-r] [-g] [-pp]
"""

import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import ImageTk, Image
import os
import sys
from detection import *
from skimage import exposure, filters, morphology, color, draw
import numpy as np
import math
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument("path", type=str, help="Path to image directory")
parser.add_argument("--color", type=str, help="color of dendrites, must be 'b' (black) or 'w' (white)")
parser.add_argument("-r", action="store_true", help="Flag to run the job")
parser.add_argument("-g", action="store_true", help="Flag to only compute graphs")
parser.add_argument("-pp", action="store_true", help="Flag to preprocess images in the specifed directory.")
parser.add_argument("--n", type=int, help="Number of CPUs to run processes over.")

args = parser.parse_args()

MODE = args.r
MAIN_PATH = args.path
BLACK_RIDGES = (True if args.color is None or args.color.lower().startswith("b") else False)
UNPROCESSED_IMAGE_PATH = MAIN_PATH
NUM_CPUS = (1 if args.n is None else args.n)

RIDGE_CACHE = {}
FRAME_CACHE = {}

STAGE_LABELS = ["Selects Cells and Raduis", "Set Ridge Thresholds", "Set Connection Points and Sources"]

MAX_REL_RANGE = np.linspace(0.01, 0.25, 250)
PROP_SRC_RANGE = np.linspace(0.001, 0.25, 250)

def mouse_to_image_coords(mouse, N, M):
    mx, my = mouse_coords
    center = int(my * N / 800), int(mx * M / 1200)
    return center


def mask_contour(mask):
    mask = mask.astype(np.uint8)
    contour = skimage.morphology.dilation(mask, out=None, shift_x=False, shift_y=False)
    contour -= mask
    return contour

def load_ridge_as_numpy(frame):
    global current_tracked_cell_index, cell_params
    cell, _, _ = cell_params[current_tracked_cell_index]
    path = "cache/" + MAIN_PATH + f"/cell{cell}/frame{frame}_ridge.png"
    if path in RIDGE_CACHE:
        return RIDGE_CACHE[path]
    img = Image.open("cache/" + MAIN_PATH + f"/cell{cell}/frame{frame}_ridge.png").convert("L")
    arr = np.array(img, dtype=np.uint8)
    RIDGE_CACHE[path] = arr
    return arr
    

def push_display_window():
    global display
    new = Image.fromarray(display).resize((1200, 800), Image.Resampling.LANCZOS)
    new = ImageTk.PhotoImage(new)
    inital_frame_label.configure(image=new)
    inital_frame_label.image = new

def center_on_cell():
    global job, current_cell_index, inital_frame_label, display, radius, current_frame
    n, m, _ = current_frame.shape
    ci, cj = job.detector.cell_centers[current_cell_index]
    left, right = max(0, ci - radius), min(m, ci + radius + 1)
    top, bottom = max(0, cj - radius), min(n, cj + radius + 1)
    sub_frame = np.copy(current_frame)[top : bottom, left : right, :]
    cx, cy = int(ci - left) , int(cj - top)
    n, m, _ = sub_frame.shape
    for offset in [-1, 0, 1]:
        cc, rr = draw.circle_perimeter(cy, cx, int(job.detector.radi[current_cell_index]) + offset)
        good_inds = np.logical_and(np.logical_and(cc >= 0, cc < n), np.logical_and(rr >= 0, rr < m))
        sub_frame[cc[good_inds], rr[good_inds], :] = [255, 255, 0]
    display = sub_frame
    push_display_window()

def change_ridge_window():
    global job, alpha, window_size, current_frame_index, current_ridge_map_grayscale
    N = len(job.detector.frames) - 1
    offset = math.floor(window_size / 2)
    left = max(0, current_frame_index - offset)
    right = min(N, current_frame_index + offset)
    window = np.array(range(left - current_frame_index, right - current_frame_index + 1))
    weights = np.exp(-alpha * window**2)
    weights /= np.sum(weights)
    maps = [load_ridge_as_numpy(current_frame_index + int(w)) for w in window]
    ridge = np.sum(maps * weights[:, np.newaxis, np.newaxis], axis=0)
    ridge = exposure.rescale_intensity(ridge, out_range=(0, 255))
    current_ridge_map_grayscale = ridge


def re_threshold():
    global current_ridge_map_rgb, current_ridge_map_grayscale, ridge_low, ridge_high, display, alpha, window_size, current_mask
    mask = filters.apply_hysteresis_threshold(current_ridge_map_grayscale, ridge_low, ridge_high)
    mask = morphology.binary_dilation(mask, footprint=morphology.disk(4))
    current_mask = mask
    temp = color.gray2rgb(2.0 * current_ridge_map_grayscale).astype(np.uint8)
    temp[mask_contour(mask) == 1, :] = [100, 100, 0]
    display = temp
    push_display_window()

def draw_markers():
    global job, current_ridge_map_grayscale, current_mask, min_max_dist, max_rel, prop_src, display, mouse_coords, current_cell_index, cell_params, current_frame, current_tracked_cell_index
    coords = feature.peak_local_max(current_ridge_map_grayscale * current_mask, min_distance=min_max_dist, threshold_rel=max_rel)
    temp = color.gray2rgb(2.0 * current_ridge_map_grayscale).astype(np.uint8)
    cell, radius, _ = cell_params[current_tracked_cell_index]
    if mouse_coords is None:
        n, m, _ = current_frame.shape
        ci, cj = job.detector.cell_centers[cell]
        left, right = max(0, ci - radius), min(m, ci + radius + 1)
        top, bottom = max(0, cj - radius), min(n, cj + radius + 1)
        center = int(cj - top), int(ci - left)
        
    else:
        mx, my = mouse_coords
        N, M, _ = temp.shape
        center = int(my * N / 800), int(mx * M / 1200)
        cell_params[current_tracked_cell_index] = (cell, radius, center)
    srcs = sorted(coords, key=lambda c: math.dist(center, c))[:math.ceil(len(coords) * prop_src)]
    for i, j in coords:
        temp[i - 1: i + 2, j - 1 : j + 2, :] = [255, 255, 0]
    for i, j in srcs:
        temp[i - 2: i + 3, j - 2 : j + 3, :] = [255, 0, 0]

    display = temp
    push_display_window()

     

def next_cell_left():
    global job, current_cell_index, current_stage, cell_params, current_tracked_cell_index
    if current_stage == 0:
        current_cell_index = (current_cell_index + 1) % len(job.detector.cell_centers)
        current_cell_label.configure(text=f"Current Cell: {current_cell_index}")
    else:
        current_tracked_cell_index = (current_tracked_cell_index + 1) % len(cell_params)
        current_cell_label.configure(text=f"Current Cell: {cell_params[current_tracked_cell_index][0]}") 
    if current_stage == 0:
        center_on_cell()
    elif current_stage == 1:
        change_ridge_window()
        re_threshold()
    else:
        draw_markers()

def next_cell_right():
    global job, current_cell_index, current_tracked_cell_index, cell_params
    if current_stage == 0:
        current_cell_index = (current_cell_index - 1) % len(job.detector.cell_centers)
        current_cell_label.configure(text=f"Current Cell: {current_cell_index}")
    else:
        current_tracked_cell_index = (current_tracked_cell_index - 1) % len(cell_params)
        current_cell_label.configure(text=f"Current Cell: {cell_params[current_tracked_cell_index][0]}") 
    if current_stage == 0:
        center_on_cell()
    elif current_stage == 1:
        change_ridge_window()
        re_threshold()
    else:
        draw_markers()

def change_detect_radius(value):
    global job, radius
    radius = 1000 - int(float(value))
    print(radius)
    center_on_cell()

def change_frame(value):
    global current_frame_index, display, current_stage, current_ridge_map_grayscale, current_ridge_map_rgb, current_mask, current_frame, cell_params, current_tracked_cell_index
    current_frame_index = math.floor(float(value))
    if current_stage == 0:
        try:
            raw_pil = Image.open(MAIN_PATH + f"/frame_{current_frame_index}.tif").convert("RGB")
        except FileNotFoundError:
            raw_pil = Image.open(MAIN_PATH + f"/frame_{current_frame_index}.png").convert("RGB")
        current_frame = np.array(raw_pil, dtype=np.uint8)
        center_on_cell()
        return
    if current_stage == 1 or current_stage == 2:
        raw_pil = Image.open("cache/" + MAIN_PATH + f"/cell{cell_params[current_tracked_cell_index][0]}/frame{current_frame_index}_ridge.png").convert("RGB")
        to_threshold = Image.open("cache/" + MAIN_PATH + f"/cell{cell_params[current_tracked_cell_index][0]}/frame{current_frame_index}_ridge.png").convert("L")
        current_ridge_map_grayscale = np.array(to_threshold, dtype=np.uint8)
        current_ridge_map_rgb = np.array(raw_pil, dtype=np.uint8)
        display = np.array(raw_pil, dtype=np.uint8)
    if current_stage == 1:
        change_ridge_window()
        re_threshold()
    if current_stage == 2:
        change_ridge_window()
        current_mask = filters.apply_hysteresis_threshold(current_ridge_map_grayscale, ridge_low, ridge_high)
        current_mask = morphology.binary_dilation(current_mask, footprint=morphology.disk(4))
        draw_markers()


def change_stage():
    global last_stage, current_stage, stage_comps, packers, current_frame_index, cell_params
    current_frame_index = min((c for c, _, _ in cell_params), key=lambda c: abs(current_cell_index - c))
    for comp in stage_comps[last_stage]:
        comp.pack_forget()
    for comp in stage_comps[current_stage]:
        comp.pack(side=tk.LEFT, padx=5, pady=5)
    change_frame(current_frame_index)


def add_cell():
    global job, current_cell_index, cell_params, radius
    new = (current_cell_index, radius, job.detector.cell_centers[current_cell_index])
    if new not in cell_params:
        cell_params.append(new)
        print(f"Added cell {new[0]} with raduis {new[1]}")
        print("Cells to track:", cell_params)

    


def change_stage_left():
    global current_stage, stage_label, last_stage
    last_stage = current_stage
    current_stage = (current_stage - 1) % 3
    stage_label.configure(text=STAGE_LABELS[current_stage] + f": {MAIN_PATH}")
    change_stage()

def change_stage_right():
    global current_stage, stage_label, last_stage
    last_stage = current_stage
    current_stage = (current_stage + 1) % 3
    stage_label.configure(text=STAGE_LABELS[current_stage] + f": {MAIN_PATH}")
    change_stage()

def change_ridge_low(value):
    global ridge_low
    ridge_low = math.floor(float(value))
    re_threshold()

def change_ridge_high(value):
    global ridge_high
    ridge_high = math.floor(float(value))
    re_threshold()

def window_size_down():
    global window_size, window_size_label
    window_size = max(3, window_size - 2)
    window_size_label.configure(text=f"Window size: {window_size}")
    change_ridge_window()
    re_threshold()


def window_size_up():
    global window_size, window_size_label
    window_size += 2
    window_size_label.configure(text=f"Window size: {window_size}")
    change_ridge_window()
    re_threshold()

def alpha_down():
    global alpha, alpha_label
    alpha = max(0.05, alpha - 0.05)
    alpha_label.configure(text=f"Alpha: {alpha:.2f}")
    change_ridge_window()
    re_threshold()


def alpha_up():
    global alpha, alpha_label
    alpha = min(1.0, alpha + 0.05)
    alpha_label.configure(text=f"Alpha: {alpha:.2f}")
    change_ridge_window()
    re_threshold()

def mmd_down():
    global min_max_dist, mmd_label
    min_max_dist = max(2, min_max_dist - 1)
    mmd_label.configure(text=f"Min Max Dist: {min_max_dist}")
    draw_markers()

def mmd_up():
    global min_max_dist, mmd_label
    min_max_dist  += 1
    mmd_label.configure(text=f"Min Max Dist: {min_max_dist}")
    draw_markers()

def change_max_rel(value):
    global max_rel
    max_rel = MAX_REL_RANGE[int(float(value))]
    draw_markers()

def change_prop_src(value):
    global prop_src
    prop_src = PROP_SRC_RANGE[int(float(value))]
    draw_markers()


def pack_first_stage():
    global current_cell_label, left_button, right_button, radius_silder, frame_slider, add_cell_button
    current_cell_label.pack(side=tk.LEFT, padx=5)
    left_button.pack(side=tk.LEFT, padx=5)
    right_button.pack(side=tk.LEFT, padx=5)
    radius_silder.pack(side=tk.LEFT, pady=20)
    frame_slider.pack(side=tk.LEFT, pady=20)
    add_cell_button.pack(side=tk.LEFT, pady=20)

def on_mouse_1_down(event):
    global mouse_coords, mouse_down, current_cell_index, current_stage, current_cell_label, cell_params
    mouse_coords = (event.x, event.y)
    if current_stage == 0:
        mouse = mouse_to_image_coords(mouse_coords, *job.detector.frames[0].shape)
        current_cell_index = min(
            range(len(job.detector.cell_centers)),
            key=lambda c: math.dist(job.detector.cell_centers[c], (mouse[1], mouse[0]))
        )
        current_cell_label.configure(text=f"Current Cell: {current_tracked_cell_index}") 
        center_on_cell()
    else:
        draw_markers()
        print("Updated sourse center:", cell_params[current_tracked_cell_index])






if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not args.r and not args.pp:
        job = DetectionJob(MAIN_PATH, BLACK_RIDGES, NUM_CPUS)
        cell_params = []


        current_cell_index = 0
        current_frame_index = 0
        current_tracked_cell_index = 0
        current_stage = 0
        last_stage = 0
        radius = 1000

        ridge_low = 1
        ridge_high = 20
        window_size = 5
        alpha = 0.1
        min_max_dist = 5
        max_rel = 0.1
        prop_src = 0.01

        current_ridge_map_grayscale = None
        current_ridge_window = None
        current_ridge_map_rgb = None
        source_center = None
        current_mask = None

        mouse_down = False
        mouse_down_coords = None
        mouse_up_coords = None
        mouse_coords = None

        selected_srcs = set()
    

        def export_job():
            job.detector.ridge_low = ridge_low
            job.detector.ridge_high = ridge_high
            job.detector.window_size = window_size
            job.detector.alpha = alpha
            job.detector.min_max_dist = min_max_dist
            job.detector.rel_max = max_rel
            job.detector.prop_src = prop_src
            job.detector.tracked_cells = cell_params

            if not os.path.exists("jobs/" + MAIN_PATH.split("/")[0]):
                os.makedirs("jobs/" + MAIN_PATH.split("/")[0], exist_ok=True)

            with open("jobs/" + MAIN_PATH + ".pkl", "wb") as infile:
                pickle.dump(job, infile)
            print("Exported Job:", MAIN_PATH, "Ridge color:", ("Black" if BLACK_RIDGES else "White"), "Tracked Cells:", [c for c, _, _ in cell_params])
            exit(0)

        def write_ridge_maps_gui():
            job.detector.read_preprocessed_frames()
            for cell, radi, _ in cell_params:
                job.detector.write_ridge_maps(cell, radi)




        root = ThemedTk(theme="equilux", themebg=True)#tk.Tk()
        root.title("demorph")

        try:
            raw_pil = Image.open(MAIN_PATH + "/frame_0.tif").convert("RGB")
        except FileNotFoundError:
            raw_pil = Image.open(MAIN_PATH + "/frame_0.png").convert("RGB")
        current_frame = np.array(raw_pil, dtype=np.uint8)
        display = current_frame

        final_pil = Image.fromarray(display).resize((1200, 800), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(final_pil)

        # Components always on screen
        export_buttom = ttk.Button(root, text="Export Job", command=export_job)
        export_buttom.pack(side=tk.LEFT, pady=10, padx=10)

        write_ridge_buttom = ttk.Button(root, text="Write Ridges", command=write_ridge_maps_gui)
        write_ridge_buttom.pack(side=tk.RIGHT, pady=10, padx=10)

        stage_label = ttk.Label(root, text=f"Select Cells and Radius: {MAIN_PATH}")
        stage_label.pack()


        inital_frame_label = ttk.Label(root, image=tk_image)
        inital_frame_label.image = tk_image  
        inital_frame_label.pack(pady=10, padx=10)
        inital_frame_label.bind("<Button-1>", on_mouse_1_down)


        left_stage_button = ttk.Button(root, text="Last Stage", command=change_stage_left)
        left_stage_button.pack(side=tk.LEFT, padx=5)

        right_stage_button = ttk.Button(root, text="Next Stage", command=change_stage_right)
        right_stage_button.pack(side=tk.LEFT, padx=5)

        current_cell_label = ttk.Label(root, text=f"Current Cell: {current_cell_index}")
        left_button = ttk.Button(root, text="Last Cell", command=next_cell_left)
        right_button = ttk.Button(root, text="Next Cell", command=next_cell_right)

        # Components for first stage

        radius_silder_label = ttk.Label(root, text="Radius:")
        radius_silder = ttk.LabeledScale(root, from_=50, to=1000)
        radius_silder.scale.configure(orient=tk.HORIZONTAL, command=change_detect_radius)
        frame_slider_label = ttk.Label(root, text="Frame:")
        frame_slider = ttk.LabeledScale(root, from_=0, to=len(job.detector.frames) - 1)
        frame_slider.scale.configure(orient=tk.HORIZONTAL, command=change_frame)
        add_cell_button = ttk.Button(root, text="Add Cell", command=add_cell)

        # Components for second stage



        low_slider_label = ttk.Label(root, text="Low:")
        low_slider = ttk.LabeledScale(root, from_=1, to=20)
        low_slider.scale.configure(orient=tk.HORIZONTAL, command=change_ridge_low)

        high_slider_label = ttk.Label(root, text="High:")
        high_slider = ttk.LabeledScale(root, from_=20, to=200)
        high_slider.scale.configure(orient=tk.HORIZONTAL, command=change_ridge_high)
   
        window_size_label = ttk.Label(root, text=f"Window Size: {window_size}")
        window_size_down_button = ttk.Button(root, text="<", command=window_size_down)
        window_size_up_button = ttk.Button(root, text=">", command=window_size_up)
        alpha_label = ttk.Label(root, text=f"Alpha: {alpha}")
        alpha_down_button = ttk.Button(root, text="<", command=alpha_down)
        alpha_up_button = ttk.Button(root, text=">", command=alpha_up)

        # Components for last stage

        mmd_label = ttk.Label(root, text=f"Min Max Dist: {min_max_dist}")
        mmd_down_button = ttk.Button(root, text="<", command=mmd_down)
        mmd_up_button = ttk.Button(root, text=">", command=mmd_up)

        max_rel_slider_label = ttk.Label(root, text="Relative Thrsh")
        max_rel_slider = ttk.LabeledScale(root, from_=0, to=249)
        max_rel_slider.scale.configure(orient=tk.HORIZONTAL, command=change_max_rel)

        prop_src_slider_label = ttk.Label(root, text="# Sources")
        prop_src_slider = ttk.LabeledScale(root, from_=0, to=249)
        prop_src_slider.scale.configure(orient=tk.HORIZONTAL, command=change_prop_src)

        stage_comps = [
            [current_cell_label, left_button, right_button, radius_silder_label, radius_silder, frame_slider_label, frame_slider, add_cell_button],
            [frame_slider_label, frame_slider, current_cell_label, left_button, right_button, low_slider_label, low_slider, high_slider_label, high_slider, window_size_label, window_size_down_button, window_size_up_button, alpha_label, alpha_down_button, alpha_up_button],
            [frame_slider_label, frame_slider, current_cell_label, left_button, right_button, mmd_label, mmd_down_button, mmd_up_button, max_rel_slider_label, max_rel_slider, prop_src_slider_label, prop_src_slider]
        ]

        for comp in stage_comps[0]:
            comp.pack(side=tk.LEFT, padx=5, pady=5)

        root.mainloop()
    elif args.pp:
        job = DetectionJob(MAIN_PATH, BLACK_RIDGES, NUM_CPUS)
        job.detector.write_preprocessed_frames()
    else:
        with open(f"jobs/{MAIN_PATH}.pkl", "rb") as infile:
            job = pickle.load(infile)
            job.detector.num_cpus = NUM_CPUS
            job.run(graphs_only=args.g)
