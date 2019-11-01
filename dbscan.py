# -*- coding: utf-8 -*-
import numpy as np
import tree

dim = 3

def DBSCAN(points, eps, MinPts):
    """
    Density-based spatial clustering of applications with noise is a data clustering algorithm 
    proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996. It is a density-based 
    clustering non-parametric algorithm: given a set of points in some space, it groups together points that 
    are closely packed together (points with many nearby neighbors), marking as outliers points that lie 
    alone in low-density regions (whose nearest neighbors are too far away).
    """
    labels = [0] * len(points)

    # Building KD-Tree.
    pointsTemp = points.tolist()
    for p_i in range(len(pointsTemp)):
        pointsTemp[p_i].append(p_i)
    kdTree = tree.KD_Tree(points=pointsTemp)
    del pointsTemp

    # For all points, grow clusters.
    clusterNum = 0
    for P in range(0, len(points)):
        if not (labels[P] == 0):
           continue

        # Get Neighboring points. 
        distanceFunction = lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(dim))
        NeighborPts= tree.getKNN(kd_node=kdTree, 
                             point=list(points[P]), 
                             k = 100, 
                             dim = dim, 
                             dist_func= distanceFunction) 
        NeighborPts.pop(0) # Remove seed point from list.

        # Remove points that lie outside radius(eps).
        nPoint = 0
        while nPoint<len(NeighborPts)-1:
            if(NeighborPts[nPoint][0]>eps):
                NeighborPts.pop(nPoint)
            nPoint+=1

        # Reject cluster as noise if it has less than threshold(MinPts) number of points.
        if (len(NeighborPts) < MinPts):
            labels[P] = -1

        # Grow cluster using current point as seed.
        else: 
           clusterNum += 1
           growCluster(points, kdTree, labels, P, NeighborPts, clusterNum, eps, MinPts)

    
    return labels


def growCluster(points, kdTree, labels, P, NeighborPts, clusterNum, eps, MinPts):
    labels[P] = clusterNum

    i = 0
    while i < len(NeighborPts)-1:    
        pointIndex = NeighborPts[i][1][-1]

        # Assign cluster number, if points neibhoring points classified as noise.
        if labels[pointIndex] == -1:
           labels[pointIndex] = clusterNum

        # Assign cluster number, if points neibhoring points unclassified(label=0).
        elif labels[pointIndex] == 0:
            labels[pointIndex] = clusterNum

            # Get K Nearest Neighbors using seed point.
            distanceFunction = lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(dim))
            PnNeighborPts= tree.getKNN(kd_node=kdTree, 
                                   point=list(points[pointIndex]), 
                                   k = 100, 
                                   dim = dim, 
                                   dist_func=distanceFunction, 
                                   return_distances=False) 
            PnNeighborPts.pop(0) # Remove seed point from list.

            # Remove points that lie outside radius(eps).
            nPoint = 0
            while nPoint<len(NeighborPts)-1:
                if(NeighborPts[nPoint][0]>eps):
                    NeighborPts.pop(nPoint)
                nPoint+=1

            # Append points if they lie within threshold(MinPts)
            if (len(PnNeighborPts) >= MinPts):
                NeighborPts.extend(PnNeighborPts)

        i += 1        

    


