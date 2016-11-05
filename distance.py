# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:37:42 2016

@author: benjamin
"""

"""
To compute distances.
"""


from scipy.spatial import distance

def EuclideanDistance(x,y):
    """
    Calculate euclidean distance between 2 points in a euclidean space of dimension n.
    x and y are either arrays or tuples.
    """
    return distance.euclidean(x,y)