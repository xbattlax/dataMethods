# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:21:53 2020

@author: Benoît Colange
"""
from sklearn.decomposition import PCA
from scipy import io
from embedder import ClassNeRV
from quality_indicators import Quality
import matplotlib.pyplot as plt

from sklearn.manifold import MDS


#Get dataset
data,labels=map(io.loadmat('globe.mat').get,['data','labels'])

#Embed dataset
#model=ClassNeRV(perplex=32,scale_out=None,tradeoff_intra=1,tradeoff_inter=0)

model = MDS(n_components=2)
pos=model.fit_transform(data,labels)

#Plot map
fig,ax=plt.subplots()
ax.scatter(*pos.T,c=plt.get_cmap('tab10')(labels.flatten()))
plt.show()
#Assess quality