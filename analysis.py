import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from network import fuzzy_fractal, sholl, plot_network, load_graph
import os
import scipy.sparse.csgraph as graph
import math
import scipy.interpolate as interpolate
import scipy.differentiate as diff
import networkx as network
import pickle
from tqdm import tqdm
import random
import multiprocessing
from skimage import exposure, filters, color
from PIL import Image
from utility import *
from itertools import zip_longest

RESULTS_PATH = "results"
TITLES = ["Number of branches", "Number of nodes", "Average branch length", "Total length", "Maximum distance from cell", "Fractal Dimension", "Sholl RI", "Leaves (paths to center node)", "Mean Degree", "RMS Length"]
DATA_COLS = "num_nodes,num_edges,mean_length,mean_squared_length,total_length,num_leaves,max_dist,fract_dim,sholl,mean_degree".split(",")

CAPTURE_WIDTH_UM = 414
CAPTURE_HEIGHT_UM = 311
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
HOURS_PER_FRAME = 0.25
UM_PER_PIXEL = CAPTURE_WIDTH_UM / IMAGE_WIDTH

def load_network(path, video, cell, frame) -> network.Graph:
    path = f"results/{path}/networks/cell{cell}/frame_{frame}_network.pkl"
    with open(path, "rb") as infile:
        return pickle.load(infile)
    
def compute_adj_and_dist_mat(G):
    A = network.adjacency_matrix(G, weight="weight").toarray()
    degree = np.sum(np.clip(A, 0, 1), axis=0)
    root = np.argmax(degree)
    D = graph.floyd_warshall(A, directed=True)
    return root, degree, A, D


def compute_features(args):
    path, cell, frame = args
    lines = []
    max_dist = 0
    G = load_graph(path, cell, frame)
    if len(G) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        root, degree, A, D = compute_adj_and_dist_mat(G)
        mean_degree = np.mean(degree)
        #root = max(G, key=lambda u: G.degree(u))
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        lengths = np.array([data["weight"] for _, _, data in G.edges(data=True)])
        leaf_nodes = [i for i in range(len(degree)) if degree[i] == 1]
        reachable_leaf_nodes = list(filter(lambda u: D[root][u] < np.inf, leaf_nodes))
        df = (fuzzy_fractal(A, D) if len(A) > 1 else 0)
        sholl_indx = sholl(G)
        if len(reachable_leaf_nodes) == 0:
            max_dist = 0
        else:
            max_dist = max(max(D[root, l] for l in reachable_leaf_nodes), max_dist)
        return [
            num_nodes,
            num_edges,
            np.mean(lengths) if len(lengths) > 0 else 0,
            np.mean(lengths ** 2) if len(lengths) > 0 else 0,
            np.sum(lengths),
            len(leaf_nodes),
            max_dist,
            df,
            sholl_indx,
            mean_degree
        ]
  

def write_feature_file(num_frames, path, cell):
    args_generator = (
        (
            path,
            cell,
            frame
        ) for frame in range(num_frames)
    )
    print(f"Writing features on {multiprocessing.cpu_count()} cores ...")
    results = []
    with multiprocessing.Pool(processes=2) as pool:
        for r in tqdm(pool.imap(compute_features, args_generator), total=num_frames):
            results.append(r)
    cols = ["num_nodes", "num_edges", "mean_length", "mean_squared_length", "total_length", "num_leaves", "max_dist", "fract_dim", "sholl", "mean_degree"]
    data = pd.DataFrame(results, columns=cols)
    path = "results/" + path + f"/cell{cell}/features.csv"
    data.to_csv(path)


def find_growth_time(video, cell):
    path = RESULTS_PATH + f"/V-{video}/networks/cell{cell}/features.csv"
    features = pd.read_csv(path)
    #ax = plt.axes()
    agg = []
    for f in ["num_edges", "num_nodes", "total_length"]:
        N = features[f]
        t = np.array(range(len(N)))
        t = t[::2]
        N = N[::2]
        s = interpolate.make_smoothing_spline(t, N)
        delta = diff.derivative(s, t).df
        sdf = interpolate.make_interp_spline(t, delta)
        ddelta = diff.derivative(sdf, t).df
        t0 = max(enumerate(ddelta), key=lambda p: p[1])[0]
        agg.append(t0)
        # ax.plot(t, N)
        # ax.vlines(t0 * 2, ymin=0, ymax=max(N))
    return 2 * math.floor(np.mean(t0))

    
def get_t0(features):
    edges = features["num_edges"]
    try:
        return next(i for i in range(len(edges)) if edges[i] >= 10)
    except StopIteration:
        return -1

def normalize(a):
    return a / np.max(a)

def aggragate(values):
    # 1. Align arrays of different lengths into a 2D matrix
    # fill_value=np.nan ensures shorter arrays don't affect the mean of longer ones
    matrix = np.array(list(zip_longest(*values, fillvalue=np.nan))).T
    
    # 2. Replace any existing zeros with NaN if you want to ignore them 
    # (Only if 0 represents "missing data" and not a literal zero value)
    matrix[matrix == 0] = np.nan
    
    # 3. Calculate mean across the cell axis (axis=0)
    # np.nanmean ignores NaNs and handles the "num_values" division automatically
    with np.errstate(all='ignore'):
        agg = np.nanmean(matrix, axis=0)
    
    # 4. Replace any remaining NaNs (where no data existed at all) with 0
    return np.nan_to_num(agg)

def plot_features_file(axes, path, label):
    features = pd.read_csv(path)
    k = 0
    for row in axes:
        for ax in row:
            data = features[DATA_COLS[k]].to_numpy(dtype=float)
            ax.plot(range(len(data)), data, label=label)
            ax.set_title(DATA_COLS[k])
            ax.legend()
            k += 1
    




def plot_network_features(result_path, cuts):
    N_b, N_j, L, L_s, L_t = [], [], [], [], []
    D_max, Df, S, leaves, D = [], [], [], [], []
    labels = []
    n_cells = 0
    
    # 1. Data Collection
    for i, path in enumerate(sorted(os.listdir(result_path))):
        video = int(path[-1])
        print(i)
        cut = cuts[i]
        folder_path = os.path.join(result_path, path)
        
        for cell_path in os.listdir(folder_path):
            if cell_path.startswith("c"):
                feature_path = os.path.join(folder_path, cell_path, "features.csv")
                if not os.path.exists(feature_path):
                    continue
                
                features = pd.read_csv(feature_path)
                t0 = get_t0(features)
                
                if t0 > -1:
                    labels.append(f"v{video} {cell_path}")
                    
                    # Specifically constrained by 'cut'
                    N_b.append(features["num_edges"][t0:cut+1].to_numpy(dtype=float))
                    N_j.append(features["num_nodes"][t0:cut+1].to_numpy(dtype=float))
                    D_max.append(features["max_dist"][t0:cut+1].to_numpy(dtype=float))
                    leaves.append(features["num_leaves"][t0:cut+1].to_numpy(dtype=float))
                    
                    # Plot from t0 to the end of the file
                    L.append(features["mean_length"][t0:].to_numpy(dtype=float))
                    L_s.append(features["mean_squared_length"][t0:].to_numpy(dtype=float))
                    L_t.append(features["total_length"][t0:].to_numpy(dtype=float))
                    Df.append(features["fract_dim"][t0:].to_numpy(dtype=float))
                    S.append(features[r"sholl"][t0:].to_numpy(dtype=float))
                    D.append(features["mean_degree"][t0:].to_numpy(dtype=float))
                    n_cells += 1

    fig, ax = plt.subplots(2, 5, figsize=(18.5, 10.5))
    axes = ax.flatten()
    
    lines = [m for m in matplotlib.lines.lineStyles.keys() if m not in ["None", " ", ""]]
    measures = [N_b, N_j, L, L_t, D_max, Df, S, leaves, D]
    titles = ["Number of branches", "Number of nodes", "Average branch length", "Total length", 
              "Maximum distance from cell", "Fractal Dimension", "Sholl", 
              "Leaves (paths to center node)", "Mean Degree", "RMS Length"]
    ylims = [None, None, (0, 40), None, None, (0.5, 2.5), (0, 10), None, (1, 2.5), (0, 300)]
    aggragate_data = []

    for k in range(len(titles)):
        a = axes[k]
        a.set_title(titles[k])
        a.set_xlabel("Hours since $t_0$")
        
        # Handle RMS Length (special case index 9)
        if k == 9: 
            for i, (l, ls) in enumerate(zip(L, L_s)):
                t = HOURS_PER_FRAME * np.arange(len(l))
                val = np.sqrt(np.abs(l - ls))
                a.plot(t, val, linestyle=random.choice(lines), alpha=0.2, linewidth=0.75)
            
            agg_l, agg_ls = aggragate(L), aggragate(L_s)
            t_agg = HOURS_PER_FRAME * np.arange(len(agg_l))
            agg_val = np.sqrt(np.abs(agg_l - agg_ls))
            a.plot(t_agg, agg_val, linewidth=2, color="gray")
            a.set_ylabel("$\mu\mathrm{m}$")
            aggragate_data.append(agg_val)
            
        elif k < len(measures):
            curr_measure = measures[k]
            scale = 1.0
            
            if titles[k] in {"Maximum distance from cell", "Average branch length"}:
                scale = UM_PER_PIXEL
                a.set_ylabel("$\mu\mathrm{m}$")
            elif titles[k] == "Total length":
                scale = UM_PER_PIXEL / 1000
                a.set_ylabel("$\mathrm{mm}$")
            
            for i, m in enumerate(curr_measure):
                # t is generated based on the length of the specific (possibly cut) array
                t = HOURS_PER_FRAME * np.arange(len(m))
                a.plot(t, m * scale, linestyle=random.choice(lines), alpha=0.2, linewidth=0.75)

            agg = aggragate(curr_measure)
            t_agg = HOURS_PER_FRAME * np.arange(len(agg))
            a.plot(t_agg, agg * scale, linewidth=2, color="gray")
            aggragate_data.append(agg * scale)

        if ylims[k] is not None:
            a.set_ylim(ylims[k])

    plt.tight_layout()
    plt.savefig(f"results_{result_path.split('/')[-1]}.png", dpi=200)
    # Save as object dtype because arrays now have different lengths
    np.save(f"aggragate_{result_path.split('/')[-1]}.npy", np.array(aggragate_data, dtype=object))
    plt.show()


def make_snapshots(video, cell, frames):
    for frame in frames:
        G = load_network(video, cell, frame)
        plot_network(G)
        plt.show()

import pickle

def plot_aggragate_data(paths):
    TITLES = [
        "Number of branches", "Number of nodes", "Average branch length", 
        "Total length", "Maximum distance from cell", "Fractal Dimension", 
        "Sholl", "Leaves (paths to center node)", "Mean Degree", "RMS Length"
    ]
    ylims = [None, None, (0, 40), None, None, (0.5, 2.5), (0, 10), None, (1, 2.5), (0, 300)]
    
    fig, ax = plt.subplots(2, 5, figsize=(18.5, 10.5))
    axes = ax.flatten()
    
    for i, full_path in enumerate(paths):
        # Extract the folder name (e.g., 'control_images') to match the filename
        path_suffix = full_path.split('/')[-1]
        filename = f"aggragate_{path_suffix}.npy"
        
        try:
            # We must use allow_pickle=True for the ragged arrays we saved
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: Could not find {filename}. Did you run plot_network_features first?")
            continue
        
        for k in range(len(TITLES)):
            if k >= len(data):
                break
                
            a = axes[k]
            curr_data = data[k]
            
            # Create time axis based on the specific length of this feature array
            t = np.arange(len(curr_data)) * HOURS_PER_FRAME
            
            color = "steelblue" if i == 0 else "coral"
            label = "Control" if i == 0 else "Fast"
            
            a.plot(t, curr_data, label=label, linewidth=1.5, color=color)
            
            a.set_title(TITLES[k])
            a.set_xlabel("Hours since $t_0$")
            
            if TITLES[k] in {"Maximum distance from cell", "Average branch length", "RMS Length"}:
                a.set_ylabel("$\mu\mathrm{m}$")
            elif TITLES[k] == "Total length":
                a.set_ylabel("$\mathrm{mm}$")
            
            if ylims[k] is not None:
                a.set_ylim(ylims[k])
            
            if k == 0:
                a.legend()

    plt.tight_layout()         
    plt.savefig("agg_results.png", dpi=300)
    plt.show()


def show_skeletons_over_time(path, cell):
    main_path = f"maps/new_algorthim/{path}/cell{cell}"
    num_frames = len(os.listdir(main_path)) // 3
    axes = plt.axes()
    for frame in range(num_frames):
        axes.clear()
        img = plt.imread(main_path + f"/frame{frame}.png")
        skel = np.loadtxt(main_path + f"/frame{frame}.txt", dtype=int)
        img[skel[:, 0], skel[:, 1]] = 1
        axes.imshow(img)
        plt.pause(0.1)

# for i, case in enumerate(["control_images", "fast_images"]):
#     for video_path in os.listdir(f"results/new_algorthim/{case}"):
#         for cell_path in os.listdir(f"results/new_algorthim/{case}/{video_path}"):
#             cell = int(cell_path[4:])
#             num_frames = len(os.listdir(f"results/new_algorthim/{case}/{video_path}/{cell_path}/networks"))
#             write_feature_file(num_frames, f"{case}/{video_path}", cell)
#             print(f"{case}: video: {video_path} cell: {cell} Done.")
              

#plot_network_features("results/new_algorthim/control_images")
#plot_network_features("results/new_algorthim/fast_images")
#plot_aggragate_data(["control_images", "fast_images"])

def write_paper_image_figures(path, cell, frames, name):
    main_path = f"cache/{path}/cell{cell}"
    for frame in frames:
        raw = Image.open(path + f"/frame_{frame}.tif")
        ridge = np.array(Image.open(main_path + f"/frame{frame}_ridge.png"))
        ridge = exposure.rescale_intensity(ridge, out_range=(0, 255))
        mask = filters.apply_hysteresis_threshold(ridge, 20, 50)
        image = Image.open(main_path + f"/frame{frame}.png")
        skel = np.loadtxt(main_path + f"/frame{frame}.txt", dtype=int)


        contours = get_contours(cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY))

        raw.convert("L").save(f"figures/{name}_raw_{frame}.png")
        masked = ridge*mask
        Image.fromarray(masked.astype(np.uint8)).convert("L").save(f"figures/{name}_ridge_{frame}.png")
        image.convert("L").save(f"figures/{name}_prep_{frame}.png")

        axes = plt.axes()
        tags = tag_skelton(skel)
        h, w = masked.shape[:2]
        axes.imshow(color.rgb2gray(masked[:, :, :3]), cmap="gray")
        for x, y in zip(skel[:, 0], skel[:, 1]):
            axes.plot(y, x, marker=("," if tags[(x, y)] == 2 else "o"), markersize=(3 if tags[(x, y)] == 2 else 1), color="red")
        axes.set_axis_off()
        axes.set_xlim(0, w)
        axes.set_ylim(h, 0)
        plt.savefig(
            f"figures/{name}_skel_{frame}.png",
            bbox_inches='tight',  
            pad_inches=0,           
            dpi=300               
        )

        plt.cla()
        plt.clf()
        axes = plt.axes()
        axes.imshow(color.rgb2gray(masked[:, :, :3]), cmap="gray")
        axes.set_axis_off()
        


        for x, y in zip(skel[:, 0], skel[:, 1]):
            axes.plot(y, x, marker="o", markersize=0.25, color="red")
        with open(f"results/{path}/cell{cell}/networks/frame_{frame}_network.pkl", "rb") as infile:
            G = pickle.load(infile)
            for u in G:
                axes.plot(u[1], u[0], marker="o", markersize=1, color="blue")

        for c in contours:
            center, radius = cv.minEnclosingCircle(c)
            circle = matplotlib.patches.Circle(center,  radius=radius, color="blue")
            axes.add_patch(circle)
        axes.set_xlim(0, w)
        axes.set_ylim(h, 0)
        plt.savefig(
            f"figures/{name}_cleaned_skel_{frame}.png",
            bbox_inches='tight',  
            pad_inches=0,           
            dpi=300               
        )

# write_paper_image_figures("control_images/V-15", 1, [0, 13, 20, 31], "control1")
# write_paper_image_figures("control_images/V-21", 2, [128, 136, 143, 154], "control2")
# write_paper_image_figures("control_images/V-25", 8, [0, 22, 40, 66], "control3")
# write_paper_image_figures("fast_images/v35", 2, [0, 22, 35, 75], "fast1")
# write_paper_image_figures("fast_images/v6", 1, [0, 8, 18, 32], "fast2")
# write_paper_image_figures("fast_images/v21", 8, [0, 12, 17, 35], "fast3")


control_cuts = [
    int((22 + 30 + 36 + 25)/4),
    7,
    30,
    133,
    70,
    int((83 + 133)/2),
    78,
    108,
    93,
    101,
    47,
    130,
    140,
    106,
    32,
    int((131+144)/2),
    int((80 + 91)/2),
    121,
    145,
    120,
    116,
    43,
    67,
    26,
    35,
    35,
    43,
    17,
    100,
    100,
    100
]

fast_cuts = [
    68,
    15,
    25,
    27,
    21,
    34,
    23,
    13,
    25,
    53,
    44,
    15,
    13,
    12,
    4,
    18,
    20,
    11,
    23,
    29,
    19,
    65,
    49,
    104,
    40,
    41,
    26,
    28,
    47,
    15,
    15,
    27,
    48,
    17,
    44,
    25,
    45,
    19
]
# path = "results/fast_images"
# for video_path in os.listdir(path):
#     for cell_path in os.listdir(path + "/" + video_path):
#         if not os.path.exists(path + "/" + video_path + "/" + cell_path + "/features.csv"):
#             num_frames = len(os.listdir(path + "/" + video_path + "/" + cell_path + "/networks"))
#             write_feature_file(num_frames, f"fast_images/{video_path}", int(cell_path[4:]))


plot_aggragate_data(["results/control_images", "results/fast_images"])
# plot_aggragate_data(["results/new_algorthim/control_images", "results/new_algorthim/fast_images"])