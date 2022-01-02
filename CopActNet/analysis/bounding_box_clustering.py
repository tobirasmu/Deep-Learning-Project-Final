# -*- coding: utf-8 -*-

from CopActNet.analysis import bounding_box_clustering
#from CopActNet import config as cfg
#from pathlib import Path, PosixPath
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_bb_centroids(file_path,num_clusters=9,plot=False):
    """
    Loads a .csv file of bounding box annotations in the www.makesense.ai
    export format. Performs K-means clustering on the bounding box sizes
    and returns the centroids as anchor box candidates.

    Parameters
    ----------
    file_path : str
        Path to .csv annotations
    num_clusters : int, optional
        Number of clusters to compute.
    plot : bool, optional
        If True, the bounding box clustering is plotted along with the centroids. The default is False.

    Returns
    -------
    centroids : float array
        list of [width,height] of bounding box size centroids

    """
    df = pd.read_csv(file_path, header=None)
    w = df[df[0]=='Helmet'][3].values
    h = df[df[0]=='Helmet'][4].values
    X = np.array([[w[i],h[i]] for i in range(len(w))])
    kmeans = KMeans(n_clusters=9,random_state=0).fit(X)
    centroids = kmeans.cluster_centers_
    if plot:
        labels = kmeans.labels_
        for i in range(10):
            plt.scatter(w[labels==i],h[labels==i],label=i)
        plt.scatter(centroids[:,0],centroids[:,1],s=100,color='k')
        #1plt.legend()
        plt.title("Size-clustering of {:d} helmet bounding boxes".format(len(w)))
        plt.xlabel("width")
        plt.ylabel("height")
        plt.show()
    return centroids


if __name__ == "__main__":
    filename = "./Annotations/labels_deeplearning-copenhagen_2021-11-09-02-43-16.csv"
    centroids = get_bb_centroids(filename,plot=True)