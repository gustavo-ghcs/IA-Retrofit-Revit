import numpy as np
import time

class DataRetriever():
    def __init__(self,tf_train,centroids,clusters):
        '''
        tf_train: tf of training data
        centroids: tf cluster centroids of training data
        clusters: data index for each cluster of training data
        '''
        self.tf_train = tf_train
        self.centroids = centroids
        self.clusters = clusters
    
    def retrieve_bf(self,datum,k=20):
        # compute tf for the data boundary
        x,y = compute_tf(datum.boundary)
        y_sampled = sample_tf(x,y,1000)
        dist = np.linalg.norm(y_sampled-self.tf_train,axis=1)
        if k>np.log2(len(self.tf_train)):
            index = np.argsort(dist)[:k]
        else:
            index = np.argpartition(dist,k)[:k]
            index = index[np.argsort(dist[index])]
        return index

    def retrieve_cluster(self,datum,k=20,multi_clusters=False):
        '''
        datum: test data
        k: retrieval num
        return: index for training data 
        '''
        # compute tf for the data boundary
        x,y = compute_tf(datum.boundary)
        y_sampled = sample_tf(x,y,1000)
        # compute distance to cluster centers
        dist = np.linalg.norm(y_sampled-self.centroids,axis=1)

        if multi_clusters:
            # more candicates
            c = int(np.max(np.clip(np.log2(k),1,5)))
            cluster_idx = np.argsort(dist)[:c]
            cluster = np.unique(self.clusters[cluster_idx].reshape(-1))
        else:
            # only candicates
            cluster_idx = np.argmin(dist)
            cluster = self.clusters[cluster_idx]

        # compute distance to cluster samples
        dist = np.linalg.norm(y_sampled-self.tf_train[cluster],axis=1)
        index = cluster[np.argsort(dist)[:k]]
        return index