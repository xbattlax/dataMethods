import json
import math
import traceback

import pandas as pd
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from scipy import io
from sklearn.manifold import MDS
from tsne import tsne
from embedder import ClassNeRV
from sklearn.decomposition import PCA
import sklearn.metrics
import dataset
import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from sklearn.cluster import KMeans
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import IntVector
from rpy2.robjects import numpy2ri

from rpy2.robjects.conversion import localconverter

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Install packages
packnames = ['clusterCrit']
utils.install_packages(StrVector(packnames))

# Load packages
clusterCrit = importr('clusterCrit')
pandas2ri.activate()


json_data={}

# sklearn2ri.activate()

def classNeRV():
    model = ClassNeRV(perplex=32, scale_out=None, tradeoff_intra=1, tradeoff_inter=0)
    data, labels = dataset.dataset1()
    pos = model.fit_transform(data, labels)
    return pos


def mds():
    model = MDS(n_components=2)
    data = dataset.dataset1()
    km = KMeans(n_clusters=3)
    km.fit(data)
    labels = km.labels_
    pos = model.fit_transform(data, labels)
    return pos


def tSNE():
    data, labels = dataSet()
    Y = tsne(data, 2, 50, 20.0)
    return Y, labels


def pca():
    model = PCA(n_components=2)
    data, labels = dataSet()
    pos = model.fit_transform(data, labels)
    return pos


def dataSet():
    data, labels = map(io.loadmat('globe.mat').get, ['data', 'labels'])
    return data, labels


# tous critere
def metricsCalcul(originDataSet, dataSet):
    metrics = clusterCrit.__dict__['_rpy2r']

    metrics = str((metrics))
    metrics = metrics.replace("\'", "\"")
    metrics = json.loads(metrics)
    # metrics.
    for m in metrics:
        if not m.startswith("_"):
            if m == "extCriteria":
                try:
                    f = getattr(clusterCrit, m)
                    print(f)
                    return f(originDataSet, dataSet, "all")
                    #print("%s : %ƒ", m, f(originDataSet, dataSet, "all"))
                except AttributeError:
                    print("Methode non disponible dans ClusterCrit \n")
                except RRuntimeError:
                    print(traceback.format_exc())
                    print("skip")

def calculate(method, iteration):
    for i in range(iteration):
        dsOriginal = dataset.dataset1()
        p2 = mds()
        km = KMeans(n_clusters=3)
        km.fit(p2)
        p2 = km.predict(p2)
        p2 = p2.flatten()

        km2 = KMeans(n_clusters=3).fit(dsOriginal)
        p1 = km2.predict(dsOriginal)
        p1 = p1.flatten()
        r_ds = IntVector(p2)
        r_dsOriginal = IntVector(p1)
        json_data[method][iteration] = metricsCalcul(r_dsOriginal, r_ds)



if __name__ == "__main__":
    dsOriginal = dataset.dataset1()

    p2 = mds()
    km = KMeans(n_clusters=3)
    km.fit(p2)
    p2 = km.predict(p2)
    p2= p2.flatten()

    km2 = KMeans(n_clusters=3).fit(dsOriginal)
    p1 = km2.predict(dsOriginal)
    p1= p1.flatten()

    print(p1)
    # ds = pd.DataFrame([ds.where(), labels]).T
    # ds = pd.DataFrame(ds, index = labels)
    # r_ds = ro.conversion.py2rpy(ds)
    # r_dsOriginal = ro.conversion.py2rpy(dsOriginal)
    r_ds = IntVector(p2)
    r_dsOriginal = IntVector(p1)

    print(type(r_ds))
    # print(p1.head())
    # print(p2.head())

    # r_ds = robjects.r('r_ds[is.nan(as.numeric(r_ds))] = NA')
    metricsCalcul(r_dsOriginal, r_ds)

    # metricsCalcul(p1, p2)
