The following files are included:
-test.py: contains example code to compute an embedding of a dataset and the quality indicators for the resulting embedding
-embedder.py: contains the code of the ClassNeRV class to construct the embedding
-quality_indicators.py: contains the code of Quality class to compute the quality indicators (trustworthiness, continuity, trustworthiness inter, continuity intra and knn gain)
-environment.yml: contains the list of dependencies for this code for the above python modules and scripts.
-globe.mat: globe dataset with hemispherical classes (open with loadmat function of module scipy.io)
-three_gaussian_clusters.mat: toy dataset with 3 Gaussian clusters used in Figure 1 of the associated paper
-digits_true.mat: subsample of 500 digits with the true labels
-digits_rand.mat: subsample of 500 digits with random labels