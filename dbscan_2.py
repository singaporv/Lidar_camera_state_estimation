# -*- coding: utf-8 -*-
import numpy
import numpy as np
import sklearn.neighbors as neighbors

def DBSCAN(D, eps, MinPts):
    labels = [0]*len(D)
    kdTree = neighbors.KDTree(D)

    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        
        NeighborPts = kdTree.query_radius(D[P].reshape(1,-1), r=eps)[0]
        if (NeighborPts.shape[0] < MinPts):
            labels[P] = -1

        else: 
           C += 1
           growCluster(D, kdTree, labels, P, NeighborPts, C, eps, MinPts)
    
    return labels


def growCluster(D, kdTree, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C

    i = 0
    while i < len(NeighborPts):    
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
           labels[Pn] = C
        
        elif labels[Pn] == 0:
            labels[Pn] = C
            
            PnNeighborPts = kdTree.query_radius(D[Pn].reshape(1,-1), r=eps)[0]
            
            if (PnNeighborPts.shape[0] >= MinPts):
                NeighborPts = np.append(NeighborPts, PnNeighborPts)

        i += 1        
    


