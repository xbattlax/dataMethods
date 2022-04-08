import random

import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd


def dataset1() :

#================================================================== 2 overlaped 1 alone==============================
    features2, clusters=make_blobs(n_samples=20000, n_features=21, centers=1, cluster_std=1.4, shuffle=True, random_state=8)
    return pd.DataFrame(features2, columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
                                          "Feature6", "Feature7", "Feature8", "Feature9", "Feature10", "Feature11",
                                          "Feature12", "Feature13", "Feature14", "Feature15", "Feature16", "Feature17",
                                          "Feature18", "Feature19", "Feature20", "Feature21"])

def dataset2():
#===================================================== 3 distinct clusters =======================================
    features, clusters = make_blobs(n_samples=20000, n_features=21, centers=3, cluster_std=0.6, shuffle=True, random_state=40 )

    return pd.DataFrame(features, columns = ["Feature1", "Feature2", "Feature3","Feature4","Feature5",
                                            "Feature6","Feature7","Feature8","Feature9","Feature10","Feature11",
                                            "Feature12","Feature13","Feature14", "Feature15","Feature16", "Feature17",
                                            "Feature18", "Feature19","Feature20", "Feature21"])

def dataset3():

#===============================================================#3 Overlaped==========================

    features1, clusters = make_blobs(n_samples=20000, n_features=21, centers=3, cluster_std=1.8, shuffle=True, random_state=22)
    return pd.DataFrame(features1, columns = ["Feature1", "Feature2", "Feature3","Feature4","Feature5",
                                            "Feature6","Feature7","Feature8","Feature9","Feature10","Feature11",
                                            "Feature12","Feature13","Feature14", "Feature15","Feature16", "Feature17",
                                            "Feature18", "Feature19","Feature20", "Feature21"])

def dataset4():
#======================================================= 1 cluster + outliers=============================================
    features, clusters = make_blobs(n_samples=20000, n_features=21, centers=1, cluster_std=0.6, shuffle=True, random_state=40 )
    new_row_values = []
    for _ in range(40):
        for _ in range (21):
            new_row_values.append(random.uniform(-15.20,15.20))
        features = numpy.vstack([features, new_row_values])
        new_row_values = []

    return pd.DataFrame(features, columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
                                          "Feature6", "Feature7", "Feature8", "Feature9", "Feature10", "Feature11",
                                          "Feature12", "Feature13", "Feature14", "Feature15", "Feature16", "Feature17",
                                          "Feature18", "Feature19", "Feature20", "Feature21"])

def dataset5():

#=====================================3 clusters + outliers===================================
    features, clusters = make_blobs(n_samples=20000, n_features=21, centers=3, cluster_std=1.8, shuffle=True,
                                 random_state=22)
    new_row_values = []
    for _ in range(40):
        for _ in range (21):
            new_row_values.append(random.uniform(-80.20,80.20))
        features = numpy.vstack([features, new_row_values])
        new_row_values = []

    return pd.DataFrame(features, columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
                                           "Feature6", "Feature7", "Feature8", "Feature9", "Feature10", "Feature11",
                                           "Feature12", "Feature13", "Feature14", "Feature15", "Feature16", "Feature17",
                                           "Feature18", "Feature19", "Feature20", "Feature21"])

