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

RESULTS_PATH = "results"
TITLES = ["Number of branches", "Number of nodes", "Average branch length", "Total length", "Maximum distance from cell", "Fractal Dimension", "Sholl", "Leaves (paths to center node)", "Mean Degree", "RMS Length"]
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
    path = "results/new_algorthim/" + path + f"/cell{cell}/features.csv"
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
    acc = np.zeros(max(len(v) for v in values))
    num_values = np.zeros(max(len(v) for v in values))
    for i in range(len(num_values)):
        for v in values:
            if i < len(v) and v[i] != 0 and not np.isnan(v[i]):
                num_values[i] += 1
    for v in values:
        v[np.isnan(v)] = 0
        acc[:len(v)] += v
    num_values[num_values == 0] = 1
    return acc / num_values

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
    




def plot_network_features(result_path):
    N_b = []
    N_j = []
    L = []
    L_s = []
    L_t = []
    D_max = []
    Df = []
    S = []
    leaves = []
    initals = []
    D = []
    n_cells = 0
    labels = []
    fig, ax = plt.subplots(2, 5)
    for path in os.listdir(result_path):
        video = int(path[-1])
        print(path)
        for cell_path in os.listdir(result_path + "/" +  path):
            print(cell_path)
            if cell_path.startswith("c"):
                feature_path = result_path + "/" + path + f"/{cell_path}/features.csv"
                if not os.path.exists(feature_path):
                    break
                print(feature_path)
                features = pd.read_csv(feature_path)
                t0 = get_t0(features)
                if t0 > -1:
                    labels.append(f"v{video} {cell_path}")
                    initals.append(t0)
                    N_b.append(features["num_edges"][t0:].to_numpy(dtype=float))
                    N_j.append(features["num_nodes"][t0:].to_numpy(dtype=float))
                    L.append(features["mean_length"][t0:].to_numpy(dtype=float))
                    L_s.append(features["mean_squared_length"][t0:].to_numpy(dtype=float))
                    L_t.append(features["total_length"][t0:].to_numpy(dtype=float))
                    D_max.append(features["max_dist"][t0:].to_numpy(dtype=float))
                    Df.append(features["fract_dim"][t0:].to_numpy(dtype=float))
                    S.append(features[r"sholl"][t0:].to_numpy(dtype=float))
                    leaves.append(features["num_leaves"][t0:].to_numpy(dtype=float))
                    D.append(features["mean_degree"][t0:].to_numpy(dtype=float))
                    n_cells += 1

  
            

    lines = list(matplotlib.lines.lineStyles.keys())
    lines.remove("None")
    lines.remove(" ")
    lines.remove("")

    # for i, l in enumerate(L):
    #     Lc = max(l)
    #     L[i] /= Lc
    #     L_t[i] /= Lc
    #     D_max[i] /= Lc

    measures = [N_b, N_j, L, L_t, D_max, Df, S, leaves, D]
    titles = ["Number of branches", "Number of nodes", "Average branch length", "Total length", "Maximum distance from cell", "Fractal Dimension", "Sholl", "Leaves (paths to center node)", "Mean Degree", "RMS Length"]
    aggragate_data = []
    k = 0
    for row in ax:
        for a in row:
            a.set_title(titles[k])
            a.set_xlabel("$t$ (Hours)")
            if k == len(measures):
                for i, (l, ls) in enumerate(zip(L, L_s)):
                    t = HOURS_PER_FRAME * np.array(range(len(l)))
                    a.plot(t, np.sqrt(np.abs(l - ls)), linestyle=random.choice(lines), alpha=0.25, label=labels[i])
                agg_l, agg_ls = aggragate(L), aggragate(L_s)
                t = HOURS_PER_FRAME * np.array(range(len(agg_l)))
                a.plot(t, np.sqrt(np.abs(agg_l - agg_ls)), linewidth=2, color="gray", label=labels[i])
                aggragate_data.append(np.sqrt(np.abs(agg_l - agg_ls)))
            else:
                if titles[k] in {"Maximum distance from cell", "Average branch length"}:
                    for i, m in enumerate(measures[k]):
                        t = HOURS_PER_FRAME * np.array(range(len(m)))
                        a.plot(t, UM_PER_PIXEL * m, linestyle=random.choice(lines), alpha=0.25, label=labels[i])
                        a.set_ylabel("$\mu\mathrm{m}$")
                    agg = aggragate(measures[k])
                    t = HOURS_PER_FRAME * np.array(range(len(agg)))
                    a.plot(t, UM_PER_PIXEL * agg, linewidth=2, color="gray")
                    aggragate_data.append(UM_PER_PIXEL * agg)
                elif titles[k] == "Total length":
                    for i, m in enumerate(measures[k]):
                        t = HOURS_PER_FRAME * np.array(range(len(m)))
                        a.plot(t, UM_PER_PIXEL / 1000 * m, linestyle=random.choice(lines), alpha=0.25, label=labels[i])
                        a.set_ylabel("$\mathrm{mm}$")
                    agg = aggragate(measures[k])
                    t = HOURS_PER_FRAME * np.array(range(len(agg)))
                    a.plot(t, UM_PER_PIXEL / 1000 * agg, linewidth=2, color="gray")
                    aggragate_data.append(UM_PER_PIXEL / 1000 * agg)
                else:
                    for i, m in enumerate(measures[k]):
                        t = HOURS_PER_FRAME * np.array(range(len(m)))
                        a.plot(t, m, linestyle=random.choice(lines), alpha=0.25, label=labels[i])

                    agg = aggragate(measures[k])
                    t = HOURS_PER_FRAME * np.array(range(len(agg)))
                    a.plot(t, agg, linewidth=2, color="gray")
                    aggragate_data.append(agg)
            k += 1

    
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout()
    plt.savefig(f"results_{result_path.split("/")[-1]}.png", dpi=200)
    np.savetxt(f"aggragate_{result_path.split("/")[-1]}.txt", np.array(aggragate_data))
    #plt.show()


def make_snapshots(video, cell, frames):
    for frame in frames:
        G = load_network(video, cell, frame)
        plot_network(G)
        plt.show()

def plot_aggragate_data(paths):
    fig, ax = plt.subplots(2, 5)
    for path in paths:
        data = np.loadtxt(f"aggragate_{path}.txt")
        k = 0
        for row in ax:
            for a in row:
                a.plot(range(len(data[k])), data[k], label=path)
                a.set_title(TITLES[k], fontsize=16)
                a.set_xlabel("$t$", fontsize=16)
                a.legend()
                k += 1
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout()         
    plt.savefig("agg_results.png", dpi=300)


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
              

# plot_network_features("results/new_algorthim/control_images")
# plot_network_features("results/new_algorthim/fast_images")
# plot_aggragate_data(["control_images", "fast_images"])