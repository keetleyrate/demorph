import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import random
from skimage.measure import label
import skimage.measure as measure
from skimage.filters import threshold_li
from tqdm import tqdm
import scipy.sparse as sparse
import scipy.sparse.csgraph as graph
import copy
import itertools
import math
import networkx as network
import pickle
from utility import *
import warnings
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
import networkx as nx

MIN_DIST_TO_CELL = 7
MIN_NODE_DIST = 5
C_EPS = 0.5
MIN_CELL_RAD = 8

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    # if not nx.is_tree(G):
    #     raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def get_contours(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    denoised = cv.GaussianBlur(image, ksize=(3, 3), sigmaX=4)
    thrsh = denoised > threshold_li(denoised)
    #thrsh = morphology.binary_erosion(thrsh, morphology.disk(4))
    contours, _ = cv.findContours(np.uint8(thrsh), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def get_blob_mask_and_cell_map(image):
    contours = get_contours(image)
    mask = np.ones(image.shape)
    cv.drawContours(mask, contours, -1, 0, -1)
    #mask = np.uint8(morphology.binary_erosion(mask, morphology.disk(16)))
    return mask

def neighbours(p, image):
    n, m = image.shape
    i, j = p
    cross = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] 
    diag = [(i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
    cands = cross + diag
    return filter(lambda p: 0 <= p[0] <= n - 1 and 0 <= p[1] <= m - 1, cands)

def get_nodes(S):
    """Return a list of sets that contain the pixel coordinates of the pixels in each node patch in S."""
    node_postions = []
    labels, num = label(S, connectivity=2, return_num=True)
    postions = [np.where(labels == i) for i in range(1, num + 1)]
    for pi, pj in postions:
        tag = S[pi[0], pj[0]]
        if tag == 3 or tag == 1:
            node_postions.append(set(zip(pi, pj)))
    return node_postions

def bfs_to_connected_nodes(S, node, node_sets):
    adjacent_nodes = []
    visited = set()
    queue = collections.deque([[random.choice(list(node_sets[node]))]])
    while len(queue) > 0:
        path = queue.popleft()
        u = path[-1]
        visited.add(u)
        for v in [v for v in neighbours(u, S) if v not in visited and S[*v] > 0]:
            if (node_index := next((i for i in range(len(node_sets)) if v not in node_sets[node] and v in node_sets[i]), None)) is not None:
                adjacent_nodes.append((node_index, len(path)))
                visited.add(v)
            else:
                queue.append(path + [v])
    return adjacent_nodes

def is_path_to_root(A, node, root):
    visited = set()
    stack = [node]
    while len(stack) > 0:
        u = stack.pop()
        if u == root:
            return True
        visited.add(u)
        for v in range(len(A[u])):
            if v not in visited and A[u][v] > 0:
                stack.append(v)
    return False




def find_nodes_to_merge(G: network.Graph, min_length):
    for u in G:
        to_merge = []
        for v in G[u]:
            if G[u][v]["weight"] <= min_length:
                to_merge.append(v)
        if len(to_merge) > 0:
            return to_merge + [u]
    return []

def merge_nodes(G: network.Graph, nodes):
    # Add new "merged" node
    m = tuple(np.mean(np.array(nodes), axis=0))
    G.add_node(m)
    # Connect the neighbours of the close neighbours to m
    for u in nodes:
        for v in G[u]:
            if v not in nodes:
                w = math.dist(m, v)
                G.add_weighted_edges_from([(m, v, w), (v, m, w)])
    # Remove u and close neighbours
    G.remove_nodes_from(nodes)

    

def is_neighbour(u, v):
    return abs(u[0] - v[0]) <= 1 and abs(u[1] - v[1]) <= 1

def node_on_edge(node, edge):
    for n in node:
        if any(is_neighbour(n, e) for e in edge):
            return True
    return False

def dfs_to_connected_nodes(node, node_sets, tags):
    adjacent_nodes = []
    visited = set()
    stack = [[random.choice(list(node_sets[node]))]]
    while len(stack) > 0:
        path = stack.pop()
        u = path[-1]
        visited.add(u)
        for v in [v for v in neighbours_no_image(u) if v not in visited and v in tags]:
            if (node_index := next((i for i in range(len(node_sets)) if v not in node_sets[node] and v in node_sets[i]), None)) is not None:
                adjacent_nodes.append((node_index, len(path)))
                visited.add(v)
            else:
                stack.append(path + [v])
    return adjacent_nodes

def add_to_node_sets(node_cell, node_sets):
    for node_set in node_sets:
        for u in node_set:
            if node_cell in neighbours_no_image(u):
                node_set.add(node_cell)
                return True
    return False

def node_close_to_cell(G, circles):
    centers = set(c for c, r in circles)
    for u in G:
        for center, r in circles:
            if u not in centers:
                if math.dist(u, center) - r < MIN_DIST_TO_CELL:
                    return u, center
    return None

def graph_from_skeleton(args):
    path, cell, frame = args
    skel = np.loadtxt("cache/" + path + f"/cell{cell}/frame{frame}.txt").astype(int)
    img = cv.imread("cache/" + path + f"/cell{cell}/frame{frame}.png")
    tags = tag_skelton(skel)
    nodes = set(cell for cell in tags.keys() if tags[cell] in {1, 3})
    
    node_sets = []

    temp = copy.deepcopy(nodes)
    while len(temp) > 0:
        node_cell = temp.pop()
        if not add_to_node_sets(node_cell, node_sets):
            node_sets.append(
                set(u for u in neighbours_no_image(node_cell) if u in tags and tags[u] == 3).union({node_cell})
            )
            temp = temp - node_sets[-1]
    
    node_locs = [
        tuple(
            map(int, np.mean(np.array(list(map(np.array, node_set))), axis=0))
        ) for node_set in node_sets
    ]

    G = network.Graph()
    for u in range(len(node_sets)):
        for v, d in dfs_to_connected_nodes(u, node_sets, tags):
            G.add_weighted_edges_from([(node_locs[u], node_locs[v], d), (node_locs[u], node_locs[v], d)])
    
    cell_conts = get_contours(img)
    circles = []
    for cont in cell_conts:
        c, r = cv.minEnclosingCircle(cont)
        if r >= MIN_CELL_RAD and cirularity(cont) >= C_EPS:
            circles.append((tuple(map(int, reversed(c))), r)) # opencv coordinates be careful
    G.add_nodes_from([c for c, _ in circles]) 

    while (r := node_close_to_cell(G, circles)) is not None:
        u, c = r
        for v in list(G[u]):
            d = G[u][v]['weight']
            G.add_weighted_edges_from([(v, c, d), (c, v, d)])
        G.remove_node(u)

    while len((nodes_to_merge := find_nodes_to_merge(G, MIN_NODE_DIST))) > 0:
        merge_nodes(G, nodes_to_merge)

    with open("results/" + path + f"/cell{cell}/networks/frame_{frame}_network.pkl", "wb") as infile:
        pickle.dump(G, infile)
            
def load_graph(path, cell, frame):
    with open("results/" + path + f"/cell{cell}/networks/frame_{frame}_network.pkl", "rb") as infile:
        return pickle.load(infile)

def plot_graph(axes, path, cell, frame):
    with open("results/" + path + f"/cell{cell}/networks/frame_{frame}_network.pkl", "rb") as infile:
        G = pickle.load(infile)
        img = cv.imread("cache/" + path + f"/cell{cell}/frame{frame}.png")
        img = skimage.color.rgb2gray(img[:, :, :3])
        axes.imshow(img, cmap="gray")
                
        for u in G:
            for v in G[u]:
                axes.plot([u[1], v[1]], [u[0], v[0]], marker="", color="blue", linewidth=0.75)

        for u in G:
            axes.plot(u[1], u[0], marker=".", color="gold", markersize=4.0)

    
        




def plot_network(G, image=None):
    #root = max(G, key=lambda u: G.degree(u))
    ax = plt.axes()
    if image is not None:
        ax.imshow(image)
    # for p in G:
    #     ax.plot(*reversed(p), marker="o", color="cadetblue")
    for u in G:
        for v in G[u]:
            ax.plot([u[1], v[1]], [u[0], v[0]], color="gray", linewidth=0.5)
    return ax


def compact_box_burn(D, l):
    boxes = []
    uncovered = set(range(len(D)))
    while len(uncovered) > 0:
        C = copy.deepcopy(uncovered)
        box = set()
        while len(C) > 0:
            u = random.choice(list(C))
            C.remove(u)
            box.add(u)
            burned = {v for v in C if D[u, v] >= l}
            C -= burned
        uncovered -= box
        boxes.append(box)
    return boxes
    


def fuzzy_fractal(A, D):
    N = len(A)
    if N == 0:
        return 0
    l_min, l_max = np.min(A, where=A != 0, initial=1e16), np.max(D, where=D != np.inf, initial=0)
    l = np.linspace(l_min, l_max , 500)
    omegas = [(D <= li).astype(np.uint8) for li in l]
    Fs = [np.exp(-D**2 / li**2) for li in l]
    ca = [1 / (N * (N - 1)) * np.sum(O * F - np.diag(np.diag(O * F))) for O, F in zip(omegas, Fs)]
    try:
        x, y = np.log(l), np.log(ca)
    except RuntimeWarning:
        x, y = np.log(l), np.log(ca)
    else:
        pass
    try:
        a, b = np.polyfit(x, y, deg=1)
    except:
        a = 0
    return a

def sierpinski(n):
    G = [[]]
    d = 1
    for _ in range(1, n + 1):
        G_prime = copy.deepcopy(G)
        for i in [j for j in range(len(G)) if len(G[j]) < 3]:
            for _ in range(3):
                G_prime.append([])
            G_prime[i].append((len(G_prime) - 1, d))
            G_prime[i].append((len(G_prime) - 2, d))
            G_prime[i].append((len(G_prime) - 3, d))
        G = G_prime
        d /= 2
    A = np.zeros((len(G), len(G)))
    for i in range(len(G)):
        for j, w in G[i]:
            A[i, j] = w
            A[j, i] = w
    return A

def fmb_covering(A, D):
    N = len(A)
    degrees = [sum(row > 0) for row in A]
    print(degrees)
    min_k, max_k = 1, max(degrees)
    print(N)
    Nb = []
    lb = []
    for k_cut in range(min_k, max_k + 1):
        hubs = [u for u in range(N) if degrees[u] >= k_cut]
        assignemnts = [min(hubs, key=lambda h: D[u, h]) for u in range(N)]
        diameters = [None for _ in range(len(hubs))]
        for h in range(len(hubs)):
            nodes_in_hub = filter(lambda u: assignemnts[u] == h, range(N))
    
            shortest_path_dists = [D[u, v] for u, v in itertools.combinations(nodes_in_hub, r=2)]
            d = (max(filter(lambda d: d < np.inf, shortest_path_dists)) if len(shortest_path_dists) > 0 else 0)
            diameters[h] = d
      
        Nb.append(len(hubs))
        lb.append(1 + sum(diameters) / len(hubs))
    plt.plot(lb, Nb)
    plt.show()


            
            

def covering_fractal(A, D):
    N = len(A)
    if N == 0:
        return 1
    
    l_min, l_max = np.min(A, where=A != 0, initial=1e16), np.max(D, where=D != np.inf, initial=0)
    print(l_min, l_max)
    l = np.linspace(l_min * 128, l_max, 500)

    boxes = [[compact_box_burn(D, li) for _ in range(8)] for li in tqdm(l)]
    ca = [np.mean([len(b) for b in boxesl]) for boxesl in boxes]
    mb = [np.mean([np.mean([len(bi) for bi in b]) for b in boxesl]) for boxesl in boxes]
    x, y = -np.log(l), np.log(ca)
    a, b = np.polyfit(x, y, deg=1)
    plt.plot(x, y, linestyle="", marker="o", markersize=1, color="gray")
    plt.plot(x, a * x + b, linestyle="--", color="coral")
    plt.xlabel(r"$-\ln(\ell)$")
    plt.ylabel(r"$\ln(n_b)$")
    plt.title(r"$d_\mathrm{covering}=" + f"{a:.4f}$")
    plt.show()

    
    x, y = np.log(l), np.log(mb)
    a, b = np.polyfit(x, y, deg=1)
    plt.plot(x, y, linestyle="", marker="o", markersize=1, color="gray")
    plt.plot(x, a * x + b, linestyle="--", color="coral")
    plt.xlabel(r"$-\ln(\ell)$")
    plt.ylabel(r"$\ln(M_b)$")
    plt.title(r"$d_\mathrm{covering}=" + f"{a:.4f}$")
    plt.show()
    #plt.savefig("covering_fractal.png", dpi=400)

def load_network(path) -> network.Graph:
    with open(path, "rb") as infile:
        return pickle.load(infile)

def intersects_circle(line, r):
    (x1, y1), (x2, y2) = line
    a = x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2
    b = -2 * x1 ** 2 + 2 * x1 * x2 - 2 * y1 ** 2 + 2 * y1 * y2
    c = x1 ** 2 + y1 ** 2 - r ** 2
    return b ** 2 - 4 * a * c > 1e-8 and (x1 ** 2 + y1 ** 2 <= r** 2 or x2 ** 2 + y2 ** 2 <= r ** 2) and not (x1 ** 2 + y1 ** 2 <= r** 2 and x2 ** 2 + y2 ** 2 <= r ** 2)

def sholl(G):
    root = max(G, key=lambda u: G.degree(u))
    translated_nodes = [np.array(u) - np.array(root) for u in G if u != root]
    if len(translated_nodes) == 0:
        return 0
    lines = [[np.array(u) - np.array(root), np.array(v) - np.array(root)] for u, v in G.edges()]
    r_min = min((np.linalg.norm(u) for u in translated_nodes))
    r_max = max((np.linalg.norm(u) for u in translated_nodes))
    radi = np.linspace(r_min, r_max, 100)
    Nr = [sum(map(lambda l: intersects_circle(l, r), lines)) for r in radi]
    try:
        primary = next(N for N in Nr if N > 0)
        return max(Nr) / primary
    except StopIteration:
        return 0
  
