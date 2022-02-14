# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:09:44 2020

@author: Benoît Colange
"""
import numpy as np
from sklearn import metrics
from scipy.linalg  import eigh
import scipy.optimize as optim
from functools import wraps
import time
import tracemalloc
    
class ClassNeRV(object):
    """ClassNeRV code. The simplest syntax to obtain a map of labeled data is: pos=ClassNeRV().fit_transform(data,labels)"""
    def __init__(self,perplex=32,scale_out=None,nb_dims=2,tradeoff_intra=1,tradeoff_inter=0,weight_intra=True,weight_inter=True,optim_method='BFGS',multiscale=True,verbose=True,trace_perf=True):
        """-perplex is the target perplexity (between 1 and N-1 with N the number of points to embed)
        -scale_out is either None to indicate that it should be equal to scale_in, a scalar, or an array of size Nx1
        -nb_dims is the dimensionality of the embedding space
        -tradeoff_intra and tradeoff_inter are the values of the tradeoff parameters (balancing recall and precision terms) for within and between class relations
        -weight_intra and weight_inter allow to remove the within or between class terms for an ablation study
        -optim_method indicate the optimization method used by scipy.optimize.minimize function
        -multiscale indicate whether to use several decreasing perplexities or only the target perplexity for optimization
        """
        self.perplex=perplex
        self.scale_out=scale_out
        self.nb_dims=nb_dims
        self.tradeoff_intra,self.tradeoff_inter=tradeoff_intra,tradeoff_inter
        self.weight_intra,self.weight_inter=weight_intra,weight_inter
        self.optim_method=optim_method
        self.multiscale=multiscale
        self.verbose=verbose
        self.trace_perf=trace_perf
    def fit(self,data,labels=None,metric_in='euclidean'):
        """Find an embedding fitting a specific dataset (in a metric space of the specified metric) with given labels.
        -data is a Nxd matrix containing the coordinates of N points in a d dimensional data space
        -labels is an optional 1-dimensional array of size N containing the classes of the points
        -metric is a string
        """
        self.nb_pts=data.shape[0]
        dist_in=self.dist(data,metric=metric_in)
        self.labels=labels
        if self.labels is None:
            if self.tradeoff_intra!=self.tradeoff_inter:
                raise Exception('Cannot use different tradeoff when labels are not specified')
            self.tradeoff=self.tradeoff_intra
        else:
            self.compute_tradeoff(labels)
        if self.perplex>self.nb_pts-1 or self.perplex<1:
            raise Exception('The perplexity is not adapted to the dataset')
        if self.trace_perf:
            tracemalloc.start()
            start=time.time()
        self.pos=self.cmds_init(dist_in)
        nb_scales=np.ceil(np.log2(self.nb_pts/self.perplex))
        if self.multiscale:
            perplex_list=np.maximum(self.nb_pts/2**np.arange(1,nb_scales+1),self.perplex)
        else:
            perplex_list=[self.perplex]
        for perplex0 in perplex_list:
            self.scale_in=self.set_scale_by_perplex(dist_in,perplex0)
            self.rate_in,self.log_rate_in=self.rate(dist_in,self.scale_in)
            res=optim.minimize(self.stress,self.pos,jac=self.grad,method=self.optim_method)
            if self.verbose:
                if res.success:
                    print('Optimization completed successfully in {nit:d} iterations of {optim_method:s} method with a stress of {stress:.2E}'.format(nit=res.nit,optim_method=self.optim_method,stress=res.fun))
                else:
                    print('Optimization failed due to:{0:s}'.format(res.message))
            self.pos=res.x
        del dist_in,self.rate_in,self.log_rate_in,self.tradeoff,self.tradeoff1m
        if self.trace_perf:
            end=time.time()
            print('Elapsed time:',end-start)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            tracemalloc.stop()
    @wraps(fit)
    def fit_transform(self,*args,**kwargs):
        """Fit the data and return the embedding positions"""
        self.fit(*args,**kwargs)
        return self.pos
    def cmds_init(self,dist_in):
        """Initialize with classical MultiDimensional Scaling (Torgerson 1952), which is the equivalent of PCA using the distance matrix"""
        gram_mat=-1/2*(np.nan_to_num(dist_in)**2-np.mean(np.nan_to_num(dist_in),axis=0,keepdims=True)-np.mean(np.nan_to_num(dist_in),axis=1,keepdims=True)+np.mean(np.nan_to_num(dist_in)))
        eig_val,eig_vect=eigh(gram_mat,eigvals=(gram_mat.shape[0]-1-(self.nb_dims-1),gram_mat.shape[0]-1))
        eig_val=np.flip(eig_val)#eigVal=eigVal[::-1]#sort in descending order (warning: not sorted by absolute value=>assume that eigenvalues are positive)
        eig_vect=np.flip(eig_vect,axis=1)#eigVect=eigVect[:,::-1]#sort in descending order
        init_pos=np.sqrt(eig_val)*eig_vect
        return init_pos
    def stress(self,pos):
        """compute ClassNeRV stress function for a given set of embedding space positions"""
        self.pos=pos
        dist_out=self.dist(self.pos,'euclidean')
        rate_out,log_rate_out=self.rate(dist_out,self.scale_out)
        rec=self.rate_in*(self.log_rate_in-log_rate_out)+rate_out-self.rate_in#recall term
        prec=rate_out*(log_rate_out-self.log_rate_in)+self.rate_in-rate_out#precision term
        stress=np.mean(np.nansum(self.tradeoff*rec+self.tradeoff1m*prec,axis=1),axis=0)#1/N times the stress from classnerv paper
        return stress
    def grad(self,pos):
        """compute gradient of ClassNeRV stress function for a given set of embedding space positions"""
        self.pos=pos
        dist_out=self.dist(self.pos,'euclidean')
        rate_out,log_rate_out=self.rate(dist_out,self.scale_out)
        arg_prime_out=self.arg_prime_out(dist_out,self.scale_out)
        grad_mixt_=self.tradeoff*(rate_out-self.rate_in)+self.tradeoff1m*rate_out*(log_rate_out-self.log_rate_in)
        forces=arg_prime_out*(rate_out*np.nansum(grad_mixt_,axis=1,keepdims=True)-grad_mixt_)/self.nb_pts#1/N times the gradient from classnerv paper
        dirs=self.pos[:,None,:]-self.pos[None,:,:]
        dirs=dirs/dist_out[...,None]
        grad=np.nansum(forces[...,None]*dirs,axis=1,keepdims=False)-np.nansum(forces[...,None]*dirs,axis=0,keepdims=False)
        return grad.flatten()
    def compute_tradeoff(self,labels):
        labels=labels.reshape((labels.size,1))
        class_community=(labels==labels.T)
        self.tradeoff=np.where(class_community,self.weight_intra*self.tradeoff_intra,self.weight_inter*self.tradeoff_inter)
        self.tradeoff1m=np.where(class_community,self.weight_intra*(1-self.tradeoff_intra),self.weight_inter*(1-self.tradeoff_inter))#1 minus the tradeoff
    @property
    def scale_out(self):
        if self._scale_out is None:
            return self.scale_in
        else:
            return self._scale_out
    @scale_out.setter
    def scale_out(self,scale_out):
        self._scale_out=scale_out
    @property
    def pos(self):
        return self._pos
    @pos.setter
    def pos(self,pos):
        """The optimization algorithm handles flattened arrays for the positions, which are here reshaped as Nxd"""
        self._pos=pos.reshape(self.nb_pts,self.nb_dims)
    @pos.deleter
    def pos(self,pos):
        del self._pos
    def dist(self,pts,metric):
        """Distances Delta_ij or D_ij computed from points xi_i or x_i (given in the pts matrix) according to the specified metric"""
        dist=metrics.pairwise_distances(pts,metric=metric)
        dist[np.arange(0,dist.shape[0]),np.arange(0,dist.shape[1])]=np.nan
        return dist
    def rate(self,dist,scale):
        """Compute the belonging rates beta_ij and b_ij"""
        if not np.isnan(dist[0,0]):
            raise Exception('Diagonal coefficients should be NaN')
        arg=0.5*(dist/scale)**2
        arg=arg-np.nanmin(arg,axis=1,keepdims=True)
        kernel=np.exp(-arg)#avoid numerical underflow of the exponential term for the softmin computation
        sum_kernel=np.nansum(kernel,axis=1,keepdims=True)
        if np.any(sum_kernel==0) or np.any(np.isnan(sum_kernel)):
            print(arg)
            raise Exception('...')
        rate=kernel/sum_kernel
        log_rate=-arg-np.log(sum_kernel)
        return rate,log_rate        
    def arg_prime_in(self,dist_in,scale_in):
        """The derivative of the argument alpha_i=Delta_ij/sigma_i with respect to sigma_i"""
        return -(dist_in/scale_in)**2/scale_in
    def arg_prime_out(self,dist_out,scale_out):
        """The derivative of the argument a_i=D_ij/s_i with respect to D_ij"""
        return dist_out/scale_out**2
    def perplex_in(self,dist_in,scale_in):
        rate_in,log_rate_in=self.rate(dist_in,scale_in)
        entrop_in=np.nansum(-rate_in*log_rate_in,axis=1,keepdims=True)
        perplex_in=np.exp(entrop_in)
        if np.any(perplex_in<1) or np.any(perplex_in>self.nb_pts-1):
            raise Exception('Invalid perplexity value')
        arg_prime_in=self.arg_prime_in(dist_in,scale_in)
        entrop_prime=np.nansum(rate_in*(log_rate_in+1)*(arg_prime_in-np.nansum(arg_prime_in*rate_in,axis=1,keepdims=True)),axis=1,keepdims=True)
        perplex_prime=perplex_in*entrop_prime
        return perplex_in,perplex_prime
    def set_scale_by_perplex(self,dist_in,perplex0,scale_init=None,error_max=10**-8):
        """Compute the input scale by perplexity"""
        self.nb_pts=dist_in.shape[0]
        scale_min,scale_max=self.vladymyrov_bounds(dist_in,perplex0)
        perplex_min,_=self.perplex_in(dist_in,scale_min)
        perplex_max,_=self.perplex_in(dist_in,scale_max)
        #Init the bounds:
        for init_count in range(1,111):#np.ceil(np.max(np.log2(scale_max/scale_min)))
            #Try new scale
            scale_min2=scale_min*2
            scale_max2=scale_max/2
            perplex_min2,_=self.perplex_in(dist_in,scale_min2)
            perplex_max2,_=self.perplex_in(dist_in,scale_max2)
            #Keep scale on the right side of limit
            sel_min=perplex_min2<perplex0
            scale_min[sel_min]=scale_min2[sel_min]
            perplex_min[sel_min]=perplex_min2[sel_min]
            sel_max=perplex_max2>perplex0
            scale_max[sel_max]=scale_max2[sel_max]
            perplex_max[sel_max]=perplex_max2[sel_max]
            if not(np.any(sel_min) or np.any(sel_max)):
                break
        if np.any(perplex_min>perplex0) or np.any(perplex_max<perplex0):
            raise Exception('Implementation error')
        #Init the scale:
        scale_in=(scale_min+scale_max)/2
        #Newton root-finding algorithm
        for newton_count in range(1,501):
            perplex_in,perplex_prime=self.perplex_in(dist_in,scale_in)
            if np.all(np.abs(perplex_in-perplex0)<error_max):
                break
            #Compute minimum slope:
            perplex_prime_inf=1.05*(np.where(perplex_in>perplex0,perplex_in-perplex0,perplex0-perplex_in))/np.maximum(np.where(perplex_in>perplex0,scale_in-scale_min,scale_max-scale_in),10**-16)
            #Refresh bounds:
            sel_min=perplex_min
            perplex_in,entrop_prime=self.perplex_in(dist_in,scale_in)
            sel_min=perplex_in<perplex0
            scale_min[sel_min]=np.maximum(scale_in[sel_min],scale_min[sel_min])
            perplex_min[sel_min]=np.maximum(perplex_in[sel_min],perplex_min[sel_min])
            sel_max=perplex_in>perplex0
            scale_max[sel_max]=np.minimum(scale_in[sel_max],scale_max[sel_max])
            perplex_max[sel_max]=np.minimum(perplex_in[sel_max],perplex_max[sel_max])
            #Newton step
            scale_in=scale_in-(perplex_in-perplex0)/np.maximum(perplex_prime,perplex_prime_inf)
            if np.any(scale_in<scale_min) or np.any(scale_in>scale_max):
                print('Scale sent out of range for {0:d} scales'.format(np.sum(np.logical_or(scale_in<scale_min,scale_in>scale_max))))
                scale_in=np.clip(scale_in,scale_min,scale_max)
        if self.verbose:
            perplex_in,_=self.perplex_in(dist_in,scale_in)
            print('Perplex set to {0:.1f} in {1:d} init iter and {2:d} newton iter with an error of at most {3:.2E}'.format(perplex0,init_count,newton_count,np.max(np.abs(perplex_in-perplex0))))
        return scale_in
    def vladymyrov_bounds(self,dist_in,perplex0):
        """Compute a lower and upper bound for the scale parameter using Vladymyrov 2013 - Entropic Affinities"""
        p=optim.newton(lambda p:2*(1-p)*np.log(self.nb_pts/2/(1-p))-np.log(np.minimum(np.sqrt(2*self.nb_pts),perplex0)),x0=0.75+0.25/3,x1=0.75+0.25*2/3)
        if p<0.75 or p>1:
            raise Exception('...')
        if not np.isnan(dist_in[0,0]):
            raise Exception('Should have diagonal NaN')
        dist_min1=np.nanmin(dist_in,axis=1,keepdims=True)
        dist_min2=np.nanmin(np.where(np.nan_to_num(dist_in)>dist_min1,dist_in,np.inf),axis=1,keepdims=True)
        dist_max=np.nanmax(dist_in,axis=1,keepdims=True)
        beta_min1=self.nb_pts/(self.nb_pts-1)*np.log(self.nb_pts/perplex0)/(dist_max**2-dist_min1**2)
        beta_min2=np.sqrt(np.log(self.nb_pts/perplex0)/(dist_max**4-dist_min1**4))
        beta_max=np.log(p/(1-p)*(self.nb_pts-1))/(dist_min2**2-dist_min1**2)
        scale_min=1/np.sqrt(2*beta_max)
        scale_max=1/np.sqrt(2*np.maximum(beta_min1,beta_min2))
        return scale_min,scale_max
