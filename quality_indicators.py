# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:18:09 2020

@author: Benoît Colange
"""

from sklearn import metrics
import numpy as np

class Quality(object):
    """Simple call with scale,trust,trust_inter=Quality(data,pos,labels).trust, to get the trustworthiness trust as a function of the scale (number of neighbours k)
    Inputs should be numpy arrays:
    -data: the data matrix of size Nxdelta (N the number of points, delta the dimensionality of the dataspace)
    -pos: the embedded points matrix of soze Nxd (N the number of points, d the dimensionality of the embedding space)
    -labels: the array of labels associated to the points of size N, or Nx1"""
    def __init__(self,data,pos,labels):
        self.nb_pts=data.shape[0]
        dist_in=metrics.pairwise_distances(data)
        self.rank_in=np.argsort(np.argsort(dist_in,axis=1),axis=1)
        dist_out=metrics.pairwise_distances(pos)
        self.rank_out=np.argsort(np.argsort(dist_out,axis=1),axis=1)
        self.labels=labels.reshape(self.nb_pts,1)
    @property
    def trust(self):
        """Return trustworthiness (trust) and trustworthiness between-class (trust_inter) as a function of the number of neighbours (scale)"""
        scale=np.arange(1,self.nb_pts-1)
        row_ind=np.repeat(np.arange(0,self.nb_pts),self.nb_pts)
        col_ind=np.argsort(self.rank_out,axis=1).flatten()
        rank_comp=self.rank_in[row_ind,col_ind].reshape((self.nb_pts,self.nb_pts))[:,1:]
        class_community=(self.labels[row_ind].flatten()==self.labels[col_ind].flatten()).reshape((self.nb_pts,self.nb_pts))[:,1:]
        penal_false_intra,penal_false_inter=np.empty(scale.shape),np.empty(scale.shape)
        for scale0 in scale:
            penal_false_intra[scale0-1]=np.sum(np.maximum(rank_comp[:,:scale0]-scale0,0)*class_community[:,:scale0])
            penal_false_inter[scale0-1]=np.sum(np.maximum(rank_comp[:,:scale0]-scale0,0)*np.logical_not(class_community[:,:scale0]))
        normalizer=np.where(scale<self.nb_pts/2,scale*self.nb_pts*(2*self.nb_pts-3*scale-1)/2,self.nb_pts*(self.nb_pts-scale)*(self.nb_pts-scale-1)/2)
        penal_false_intra,penal_false_inter=penal_false_intra/normalizer,penal_false_inter/normalizer
        trust=1-(penal_false_intra+penal_false_inter)
        trust_inter=1-penal_false_inter
        return scale,trust,trust_inter
    @property
    def cont(self):
        """Return continuity (cont) and continuity within-class (trust_intra) as a function of the number of neighbours (scale)"""
        scale=np.arange(1,self.nb_pts-1)
        row_ind=np.repeat(np.arange(0,self.nb_pts),self.nb_pts)
        col_ind=np.argsort(self.rank_in,axis=1).flatten()
        rank_comp=self.rank_out[row_ind,col_ind].reshape((self.nb_pts,self.nb_pts))[:,1:]
        class_community=(self.labels[row_ind].flatten()==self.labels[col_ind].flatten()).reshape((self.nb_pts,self.nb_pts))[:,1:]
        penal_missed_intra,penal_missed_inter=np.empty(scale.shape),np.empty(scale.shape)
        for scale0 in scale:
            penal_missed_intra[scale0-1]=np.sum(np.maximum(rank_comp[:,:scale0]-scale0,0)*class_community[:,:scale0])
            penal_missed_inter[scale0-1]=np.sum(np.maximum(rank_comp[:,:scale0]-scale0,0)*np.logical_not(class_community[:,:scale0]))
        normalizer=np.where(scale<self.nb_pts/2,scale*self.nb_pts*(2*self.nb_pts-3*scale-1)/2,self.nb_pts*(self.nb_pts-scale)*(self.nb_pts-scale-1)/2)
        penal_missed_intra,penal_missed_inter=penal_missed_intra/normalizer,penal_missed_inter/normalizer
        cont=1-(penal_missed_intra+penal_missed_inter)
        cont_intra=1-penal_missed_intra
        return scale,cont,cont_intra
    @property
    def knn_gain(self):
        """Return k-nn gain (gain) as a function of the number of neighbours (scale)"""
        perm_in=np.argsort(self.dist_in,axis=1)[:,1:]
        intra_in=np.cumsum(self.labels.flatten()[perm_in]==self.labels.reshape(self.nb_pts,1),axis=1)/np.arange(1,self.nb_pts)[None,:]
        perm_out=np.argsort(self.dist_out,axis=1)[:,1:]
        intra_out=np.cumsum(self.labels.flatten()[perm_out]==self.labels.reshape(self.nb_pts,1),axis=1)/np.arange(1,self.nb_pts)[None,:]
        scale=np.arange(1,self.nb_pts)
        gain=np.mean(intra_out-intra_in,axis=0)
        return scale,gain