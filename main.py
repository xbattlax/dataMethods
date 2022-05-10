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

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

from rpy2.robjects.conversion import localconverter


crit = ["Czekanowski_Dice", "Folkes_Mallows", "Hubert","Jaccard", "Kulczynski", "McNemar","Phi","Precision","Rand","Recall" ,"Rogers_Tanimoto", "Russel_Rao", "Sokal_Sneath1", "Sokal_Sneath2"]


utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Install packages
packnames = ['clusterCrit']
utils.install_packages(StrVector(packnames))

# Load packages
clusterCrit = importr('clusterCrit')
pandas2ri.activate()


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
    return Y


def pca():
    model = PCA(n_components=2)
    data, labels = dataSet()
    pos = model.fit_transform(data, labels)
    km = KMeans(n_clusters=3)
    km.fit(data)
    p2 = km.predict(pos)
    p2 = p2.flatten()
    print(p2)
    return p2


def dataSet():
    data, labels = map(io.loadmat('globe.mat').get, ['data', 'labels'])
    return data, labels

def kmeans(data, labels):
    km = KMeans(n_clusters=3)
    km.fit(data, labels)


# tous critere
def metricsCalcul(originDataSet, dataSet):
    data_j={}
    f = getattr(clusterCrit, "extCriteria")
    for m in crit:
        try:
            res = f(originDataSet, dataSet, m)
            data_j[m] = res[0][0]
        except AttributeError:
            print("Methode non disponible dans ClusterCrit \n")
        except RRuntimeError:
            print(traceback.format_exc())
            print("skip")
    return data_j

def calculate(method, iteration):
    json_data={}
    for i in range(iteration):
        dsOriginal = dataset.dataset1()
        p2 = method()
        #km = KMeans(n_clusters=3)
        #km.fit(p2)
        #p2 = km.predict(p2)
        #p2 = p2.flatten()
        print(p2)
        km2 = KMeans(n_clusters=3).fit(dsOriginal)
        p1 = km2.predict(dsOriginal)
        p1 = p1.flatten()
        r_ds = IntVector(p2)
        r_dsOriginal = IntVector(p1)
        json_data[i] = metricsCalcul(r_dsOriginal, r_ds)
    return json_data


if __name__ == "__main__":
    method = [mds, pca, tSNE, classNeRV]
    json_data= {}
    for f in method:
        name=f.__name__.__str__()
        res= calculate(f, 1)
        print(res)
        json_data[name]=res
    #calculate("mds", 10)
    json.dump(json_data, open("mds.json", "w"))


    # metricsCalcul(p1, p2)
