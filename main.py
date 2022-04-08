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
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Install packages
packnames = 'ClusterCrit'
utils.install_packages(StrVector(packnames))

# Load packages
clusterCrit = importr('ClusterCrit')


def classNeRV():
    model = ClassNeRV(perplex=32, scale_out=None, tradeoff_intra=1, tradeoff_inter=0)
    data, labels = dataset.dataset1()
    pos = model.fit_transform(data, labels)
    return pos


def mds():
    model = MDS(n_components=2)
    data, labels = dataSet()
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
def metricsCalcul(metrics, originDataSet, DataSet):
    for m in metrics:
        try:
            f = getattr(clusterCrit, m)
            print("%s : %ƒ", m, f(originDataSet, DataSet))
        except AttributeError:
            print("Methode non disponible dans ClusterCrit \n")


if __name__ == "__main__":
    dsOriginal = dataset.dataset1()
    ds = classNeRV()
    ds = sklearn.kMean(ds)
    metricsCalcul([''], dsOriginal, ds)