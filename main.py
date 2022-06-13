import json
import traceback

from rpy2.rinterface_lib.embedded import RRuntimeError
from scipy import io
from sklearn.manifold import MDS
from tsne import tsne
from embedder import ClassNeRV
from sklearn.decomposition import PCA
import dataset
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from sklearn.cluster import KMeans
from rpy2.robjects.vectors import IntVector

import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


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
    data = dataSet()
    km = KMeans(n_clusters=2)
    km.fit(data)
    labels = km.labels_
    pos = model.fit_transform(data, labels)
    p2 = km.predict(pos)
    return p2


def mds(ds ):
    model = MDS(n_components=2)
    data = ds()
    km = KMeans(n_clusters=3)
    km.fit(data)
    labels = km.labels_
    pos = model.fit_transform(data, labels)
    return pos


def tSNE(ds):
    data = ds()
    Y = tsne(data, 2, 50, 20.0)

    km = KMeans(n_clusters=3)
    km.fit(data)
    return Y


def pca(ds):
    model = PCA(n_components=2)
    data = ds()
    km = KMeans(n_clusters=3)
    km.fit(data)
    labels = km.labels_
    pos = model.fit_transform(data, labels)

    return pos


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
            data_j[m] = round(res[0][0], 2)
        except RRuntimeError:
            print(traceback.format_exc())
            print("skip")
    return data_j

def calculate(method, ds):

    json_data={}
    for d in ds:
        dsOriginal = d()
        p2 = method(d)

        km2 = KMeans(n_clusters=3).fit(dsOriginal)
        p1 = km2.predict(dsOriginal)
        p1 = p1.flatten()
        p2 = p2.flatten()

        r_dsOriginal = IntVector(p1)
        r_ds = IntVector(p2)
        json_data[d.__name__] = metricsCalcul(r_dsOriginal, r_ds)
    return json_data


if __name__ == "__main__":
    method = {"mds":mds, "tSNE":tSNE, "pca":pca}
    name = ["mds", "tSNE", "pca"]
    ds = [dataset.dataset1, dataset.dataset2, dataset.dataset3]
    json_data= {}
    for k in method:
        name=k
        res= calculate(method[k], ds)
        print(res)
        json_data[name]=res

    json.dump(json_data, open("mds.json", "w"))

    newDict = {}

    pdObj = pd.read_json("mds.json")
    for i in pdObj:
        s = pdObj[i].to_dict()
        for j in s:
            row = i + ' ' +j
            #print(row)
            newDict[row] = s[j]

    print(newDict)
    pdDs = pd.DataFrame.from_dict(newDict, orient="index")
    with open("res.csv", "w") as f:
        f.write(pdDs.to_csv())




