#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from sklearn.cluster import KMeans


def clustering_params(data, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.predict(data)
    return labels, kmeans.cluster_centers_


def dimension_reduction(data, dim=2):
    if len(data.shape) <= dim:
        return data
    from sklearn.manifold import TSNE
    low_dim_data = TSNE(n_components=dim).fit_transform(data)
    return low_dim_data


def visualize(data, labels, dim=2, name='test'):
    """

    Args:
        data_with_labels:
        [xx, yy, zz, label], centers

    Returns: None

    """
    low_dim_data = dimension_reduction(data, dim=dim)
    from matplotlib import pyplot as plt
    for kind in range(np.max(labels)):
        d = low_dim_data[labels==kind]
        plt.scatter(d[:, 0], d[:, 1], s=3)
    plt.title(name)
    plt.savefig(os.path.join('pngs', name+'.png'))
    plt.close()


for l in np.arange(5, 100, 5):
    test_data_path = '../data/linear/%d.csv' % l
    import pandas as pd
    df = pd.read_csv(test_data_path)
    data = np.array(df)[:, 1:]
    labels, centers = clustering_params(data, n_clusters=10)

    visualize(data[:, 0:2], labels, dim=2, name='01-'+str(l))
    visualize(data[:, 2:4], labels, dim=2, name='23-'+str(l))



