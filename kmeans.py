# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:51:31 2016

@author: Benjamin CHICHE, Remy LEROY

Implementation of k-means algorithm that is the most popular heuristic for solving the k-means problem.

This algorithm is tested on several datasets:
    - Iris flower dataset
    - The MNIST database of handwritten digits
    - Set of data generated randomly according to several probability distribution. These data could be difficult to distinguish if real clusters overlap
    
In the following, "points" means "data".

Some measurements help to evaluate clustering :
    - Numbers of sample in each cluster obtained compered to real number of sample in each cluster (if the ground truth is known)
    - Error rates : division of number of data attributed to wrong cluster by total number of data (if the ground truth is known)


    
References:

    - "An Efficient k-Means Clustering Algorithm: Analysis and Implementation" -
                 Tapas Kanungo, David M. Mount ,Nathan S. Netanyahu, Christine D. Piatko,	Ruth Silverman, Angela Y. Wu    

"""
import time
import random
import distance
import sys
import io2
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

global counter 

counter = 0

def affectation(centroids, points):
    """
    To affect data (points) to centroids of clusters to form new cluster
    
    @type centroids: list of lists
    @param centroids: list of lists containing coordinates of centroids

    @type points: list of lists
    @param points: list of lists containing coordianates of data (points)

    @type clusters[:]: list of lists of lists
    @return clusters[:]: list of lists of lists containing groups of points. Each group is equivalent to a cluster.     
    
    """
    clusters = []
    for i in range(len(centroids)):
        clusters.append([centroids[i]])
        
        
    for point in points:
        
        distance_to_centroid = distance.EuclideanDistance(point, centroids[0])
        count = 0

        index = 0
        for centroid in centroids[1:]:

            
    
            distance_min = distance_to_centroid 
            distance_to_centroid = distance.EuclideanDistance(point, centroid)
            
            if distance_min > distance_to_centroid:
                count += 1
                distance_min = distance_to_centroid
                index = count
                
                
            else:
                count += 1

        clusters[index].append(point)

    return clusters[:]


def new_centroids(clusters):
    """
    To update centroids of clusters after formation of new clusters
    
    @type clusters: list of lists of lists
    @param clusters: list of lists of lists containing groups of points. Each group is equivalent to a cluster.     

    @type centroids[:]: list of lists 
    @return centroids[:]: list of lists containing coordinates of centroids 
    
    """
    centroids = []
    for cluster in clusters :
        centroid = new_mean(cluster)
        centroids.append(centroid)
        
    return centroids[:]
    
        
def new_mean(cluster):
    """
    To calculate new mean of a cluster

    @type cluster: list of lists
    @param cluster: list containing coordinates of data belonging to a same cluster

    @type G[:]: list  
    @return G[:]: list containing coordinates of new mean of the cluster     
    
    """
    
    dimension = len(cluster[0])
    G = []
    denominateur = len(cluster)
    for i in range (dimension):
        coord = 0
        for point in cluster:
            coord += point[i]
        G.append(coord/denominateur)
    return G[:]    
    
def kMeans(ini_centroids, points, max_iter):
    """
    To applicate k-means algorithm starting from given centroids among given points (data)

    @type ini_centroids: a list of lists, each list describing coordinates of given centroids
    @param ini_centroids: initial given centroids of clusters
    
    @type points: a list of lists, each list describing data, i.e coordinates of given points 
    @param points: given data (points) which wanted to be clustered
    
    @type max_iter: int
    @param max_iter: maximal number of iterations

    @type centroids[:], clusters[:]: tuple of lists 
    @return centroids[:], clusters[:]: list containing coordinates of final centroids and list of lists containing information about clusters formed (each list contains points belonging in a same cluster)       
    
    """
    centroids = ini_centroids[:]
    for i in range (max_iter): 
        global counter
        counter += 1
        clusters = affectation(centroids, points)
        previous_centroids = centroids[:]
        centroids = new_centroids(clusters)
        
        # To verify another condition than number of iterations to know whether situation converged
        if variations(previous_centroids,centroids,0) :

            break
        
    return centroids[:], clusters[:]    

        
def Forgy_initialization(points,k):
    """
    The Forgy method that randomly chooses k centroids to begin k-means algorithm
    
    @type points: a list of lists, each list describing data, i.e coordinates of given points 
    @param points: given data (points) which wanted to be clustered    
    
    @type k: int  
    @param k: number of clusters that has to be input
    
    @type points[:k]: list of list
    @return points[:k]: list of chosen k centroids to start k-means algorithm
    
    """
    random.shuffle(points)
    return points[:k]
        
def variations(previous_centroids,centroids, epsilon):
    """
    To verify whether clustering converged    
    
    @type previous_centroids: list of list
    @param previous_centroids: list of centroids before update
    
    @type centroids: list of list
    @param centroids: list of updated centroids
    
    @type epsilon: float
    @param epsilon: parameter that is compared to the difference between previous_centroids and centroids

    @type sqrt(var) < epsilon: bool
    @return sqrt(var) < epsilon: indicates whether clustering converged or not 
    
    """        
    
    var = 0
    for i in range (len(centroids)):
        var += (distance.EuclideanDistance(centroids[i],previous_centroids[i]))**2
    return sqrt(var) < epsilon    
                
    
def main():
    """
    """
    # enter data to cluster
    # for example, sys.argv[1] could be "iris.data.txt"
    arg =  sys.argv[1] 
    #arg =  "iris.data.txt"
    
    if arg == "iris.data.txt":
            
        # Here we test on IRIS data    
        # io.py is replaced by io2.py, adapted to the format of "iris.data.txt".     
        points = io2.read_data(arg, ignore_last_column = True)
        max_iter = 100
        k = 3
        centroids = Forgy_initialization(points,k)
        t1 = time.clock()
        C, clusters = kMeans(centroids,points,max_iter)
        t2 = time.clock()
        for cluster in clusters:
            print(len(cluster))
        print(counter)
        print(t2 - t1)
        
    elif arg == "MNIST":    
        # Test on MNIST dataset
        from sklearn.datasets import load_digits
        dataset = load_digits()
        data = dataset['data']
        points = []
        for i  in range (data.shape[0]):
            points.append(data[i].tolist())
        
        
        max_iter = 100
        k = 10
        centroids = Forgy_initialization(points,k)
        t1 = time.clock()        
        C, clusters = kMeans(centroids,points,max_iter)
        t2 = time.clock()        
        for cluster in clusters:
            print(len(cluster))
        print(counter)    
        print (t2 - t1)
    else : 
        np.random.seed(4711)  # for repeatability of this tutorial
        a = np.random.multivariate_normal([5, 0], [[3, 1], [1, 4]], size=[20,])
        b = np.random.multivariate_normal([0,0], [[3, 1], [1, 4]], size=[20,])  
        # c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[20,])
        # d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[20,])
        # e = np.random.multivariate_normal([0, 100], [[100, 1], [1, 100]], size=[20,])
        # X2 = np.concatenate((a,b, c, d, e),)
        X2 = np.concatenate((a,b),)
        # plt.scatter(X2[:,0], X2[:,1])
        # plt.show()    
        a = a.tolist()
        b = b.tolist()
        points = X2.tolist()    
        
        points_labels = []
        for i in a:
            points_labels.append(('a',i))
            
        for j in b:
            points_labels.append(('b',j))    
            
        max_iter = 100
        k = 2
        centroids = Forgy_initialization(points,k)
        C, clusters = kMeans(centroids,points,max_iter)
        
        for cluster in clusters:
            print(len(cluster))
        # a = np.random.multivariate_normal([5, 0], [[3, 1], [1, 4]], size=[20,])
        b = np.random.multivariate_normal([0,0], [[3, 1], [1, 4]], size=[20,])  
        # c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[20,])
        # d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[20,])
        e = np.random.multivariate_normal([0, 100], [[100, 1], [1, 100]], size=[20,])
        X2 = np.concatenate((b, e),)
        # plt.scatter(X2[:,0], X2[:,1])
        # plt.show()    
    
        points = X2.tolist()    
    
            
        max_iter = 100
        k = 2
        centroids = Forgy_initialization(points,k)
        t1 = time.clock()
        C, clusters = kMeans(centroids,points,max_iter)
        t2 = time.clock()        
        for cluster in clusters:
            print(len(cluster))
        print(counter)    
        print(t2-t1)

if __name__ == "__main__":
    main()            
    
    
    

        
        
                
            