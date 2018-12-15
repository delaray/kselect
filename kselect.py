# *********************************************************************************************
# Selecting the Best K for Kmeans
# *********************************************************************************************

import math 
from numpy import sqrt
from numpy.random import randint
import numpy as np
import random

from scipy.spatial import distance
import pandas as pd
import operator
        
# *********************************************************************************************
# Bayesian Information Criterion Calculation
# *********************************************************************************************

# ------------------------------------------------------------------------------------------
# Compute Cluster Variance
# ------------------------------------------------------------------------------------------

def compute_cluster_variance (cluster_df, center, k):
    cluster_size = cluster_df.shape[0]
    # print ("Cluster Size: " + str(cluster_size) + " ,K: " + str(k))
    squared_distances = []
    for index, row in cluster_df.iterrows():
        point = (row['x'], row['y'])
        d = distance.euclidean(point, center)
        squared_distances.append(d**2)
        return sum (squared_distances) / cluster_size
        # return sum (squared_distances) 
        
# ------------------------------------------------------------------------------------------
# Compute Model Variance
# ------------------------------------------------------------------------------------------

# This computes the sum of the squares of distances of each point from the center of
# it's cluster divided by the total number of points.

def compute_model_variance (clusters_df, centers):
    k = len(centers)
    N = clusters_df.shape[0]
    variances = []
    clusters = clusters_df['cluster'].unique()
    for cluster_label in clusters:
        cluster_df = clusters_df.loc[clusters_df['cluster'] == cluster_label]
        center = centers[cluster_label]
        variances.append(compute_cluster_variance(cluster_df, center, k))
    return sum(variances) 
    #return sum(variances) / N

# ------------------------------------------------------------------------------------------
# Single Datapoint Density
# ------------------------------------------------------------------------------------------

def vector_norm (vector):
    return sqrt(sum([x**2 for x in vector]))

def vector_diff (v1, v2):
    return list(map(operator.sub, v1, v2))

# ------------------------------------------------------------------------------------------

# This computes P(uj | M, sigma2) where uj is a datapoint, M contains all the means
# (i.e. cluster centers) and sigma squared is the overall variance.

def data_point_density (point, center, variance):
    mfactor = 1.0 / (sqrt (2 * np.pi * variance)) if variance != 0 else 0
    uj = point
    ukj = center
    norm = vector_norm(vector_diff (uj, ukj))
    exponent = - 0.5 * (norm**2 / variance) if variance != 0 else 0
    density = mfactor * (np.e ** exponent)
    return density

# -------------------------------------------------------------------------------------------------
# Data Set Density
# -------------------------------------------------------------------------------------------------

def data_model_density (cluster_df, centers):

    # Number of datapoints in the cluster 
    N = cluster_df.shape[0]
    
    # Compute the variance of the model 
    variance = compute_model_variance (cluster_df, centers)

    # Sum the log of the individual densities of the data points.
    result = 0
    for index, row in cluster_df.iterrows():
        point = [row['x'], row['y']]
        center = centers[int(row['cluster'])]
        density = data_point_density(point, center, variance)
        if density != 0:
            result = result + np.log(density)

    return result, variance

# -------------------------------------------------------------------------------------------------
# Bayesian Information Criteria
# -------------------------------------------------------------------------------------------------

def BIC (cluster_df, centers):
    k = len(centers)
    Q = len(centers[0])
    N = cluster_df.shape[0]
    density, variance = data_model_density(cluster_df, centers)
    k_factor = k  * np.log(N)
    bic = density - ((((k+1) * Q) / 2) * np.log(N))
    return bic, density, variance
    #return  k_factor - (2 * density) , density, variance

# -------------------------------------------------------------------------------------------------

def extract_stats(stats):
    bics = [[x[0], x[1][0]] for x in stats]
    densities = [[x[0], x[1][1]] for x in stats]
    variances = [[x[0], x[1][2]] for x in stats]
    return bics, densities, variances

# -------------------------------------------------------------------------------------------------
# Find Maximim Gap in BICs
# -------------------------------------------------------------------------------------------------

def scale_points (points):
    min_x = np.min([p[0] for p in points])
    max_x = np.max([p[0] for p in points])
    min_y = np.min([p[1] for p in points])
    max_y = np.max([p[1] for p in points])
    new_points = [[x, (x-min_x)/(max_x-min_x), (y-min_y)/(max_y-min_y)] for x,y in points]
    return new_points
        
# *********************************************************************************************
# Elbow Method For Selecting the Best K for KMeans.
# *********************************************************************************************

# Input: Bics is a vector of vectors of the form [k bic].

# Methodology: Find the largest angle less 45 degrees between succes pairs
# of points (k, bic)

# Return the boundaries of range of K values.

def find_elbow (bics):
    bic_dict = dict(bics)
    bics = scale_points(bics)
    pairs = [[e1, e2] for e1, e2 in zip(bics[:-1],bics[1:])]
    slopes = [[p[0], p[1], np.abs(p[1][2]-p[0][2]) / np.abs(p[1][1]-p[0][1])] for p in pairs]
    angles = [[p1 , p2, np.degrees(np.arctan(slope))] for p1, p2, slope in slopes]
    found = angles[0]
    for angle in angles:
        if angle[2] > 45:
            found = angle
        else:
            k1 = found[0][0]
            b1 = bic_dict[k1]
            k2 = found[1][0]
            b2 = bic_dict[k2]
            return [[k1, b1], [k2, b2]]

# -------------------------------------------------------------------------------------------------

# This chooses largest gap in BICs.

# Input: Bics is a vector of vectors of the form [k bic].
# Output: A vector of [k1 b1] [k2 b2] and (- b2 b1).

def find_max_gap (bics):
    bics = scale_points(bics)
    successive_pairs = [[e1, e2] for e1, e2 in zip(bics[:-1],bics[1:])]
    sort_key = lambda x : np.abs(x[1][2] - x[0][2])
    successive_pairs.sort(key=sort_key)
    successive_pairs = successive_pairs[::-1]
    print (successive_pairs)
    return successive_pairs[0]

#*******************************************************************************************
# MISCELLANEOUS FUNCTIONS
#*******************************************************************************************

# ------------------------------------------------------------------------------------------
# Generates a grid of random real numbers.
# ------------------------------------------------------------------------------------------

def generate_random_grid (K, size, min=0, max=1000, margin=5):
    random.seed()
    N = int(sqrt(K))
    points = []
    points_per_box = int(size/K)
    box_size = int((max-min) / N)
    
    for i in range(N):
        for j in range(N):
            low_x = min + margin + (i * box_size)
            high_x = min + ((i+1) * box_size) - margin
            low_y = min + margin + (j * box_size)
            high_y = min + ((j+1) * box_size) - margin
            for k in range (points_per_box):
                px = random.uniform(low_x, high_x)
                py = random.uniform(low_y, high_y)
                points.append([px, py])
    df = pd.DataFrame(points, columns=['x', 'y'])
    return df

# ------------------------------------------------------------------------------------------

def powers_of_two (maximum, minimum=2):
    results = [minimum*minimum]
    power = 2
    while results[-1] < maximum/2:
        power += 1
        results.append (2**power)
    return results[:-1]

#*******************************************************************************************
# End
#*******************************************************************************************
