# -*- coding: utf-8 -*-
import numpy as np
import heapq

def KD_Tree(points, i=0):
    """
    Build KD Tree from given points.
    """

    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % 3
        half = len(points) >> 1         #Bitwise division
        return (KD_Tree(points[: half], i), KD_Tree(points[half + 1:], i), points[half])

    elif len(points) == 1:
        return (None, None, points[0])

def getKNN(kd_node, point, k, dim, dist_func, return_distances=False, i=0, heap=None):
    """
    Get K Nearest Neighbors.
    """
    root = not heap  # Set to pass it through the first element
    if root:         # Initially always false
        heap = []       # Create a new heap tree at the very beginning

    if kd_node:         
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]

        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))

        i = (i + 1) % 3
        # Goes into the left branch, and then the right branch if needed
        getKNN(kd_node[dx < 0], point, k, dim, dist_func, return_distances, i, heap)
        # -heap[0][0] is the largest distance in the heap
        if dx * dx < -heap[0][0]:  
            getKNN(kd_node[dx >= 0], point, k, dim, dist_func, return_distances, i, heap)

    if root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors

