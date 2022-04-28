import json
import math

import pandas as pd
from rpy2 import robjects
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

from rpy2.robjects.conversion import localconverter


utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)


# Install packages
packnames = ['clusterCrit']
utils.install_packages(StrVector(packnames))

# Load packages
clusterCrit = importr('clusterCrit')
pandas2ri.activate()
#sklearn2ri.activate()

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
def metricsCalcul( originDataSet, DataSet):
    metrics = clusterCrit.__dict__['_rpy2r']
    metrics=str((metrics))
    metrics = metrics.replace("\'", "\""                       )
    metrics=json.loads(metrics)
    #metrics.
    for m in metrics:
        if not m.startswith("_"):
            try:
                f = getattr(clusterCrit, m)
                print("%s : %ƒ", m, f(originDataSet, DataSet))
            except AttributeError:
                print("Methode non disponible dans ClusterCrit \n")


if __name__ == "__main__":
    dsOriginal = dataset.dataset1()
    ds = mds()
    km = KMeans(n_clusters=3)
    km.fit(ds)
    labels = km.labels_
    #ds = pd.DataFrame([ds.where(), labels]).T
    ds = pd.DataFrame(ds, index = labels)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_ds = ro.conversion.py2rpy(ds)

    #r_ds = robjects.r('r_ds[is.nan(as.numeric(r_ds))] = NA')
    metricsCalcul( dsOriginal, r_ds)