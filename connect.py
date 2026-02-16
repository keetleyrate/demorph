"""
File: connect.py
Author: Keetley Rate
Role: Research Intern OIST
Date: February 16, 2026
Description:
    Implements a Lowest-Cost-First Search (LCFS) algorithm to connect 
    disparate points along "ridges" in a grid. Used for 
    topological network reconstruction from an image.
"""

import heapq
import copy
from utility import *

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


