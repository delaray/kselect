# *********************************************************************************************
# Selecting the Best K for Kmeans
# *********************************************************************************************

# System modules
import os
import sys
import operator

# Math
import math 
import random
import numpy as np
from numpy import sqrt
from numpy.random import randint
from numpy import round, abs

# Data Frames
import pandas as pd

# ML
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn import mixture

# Pyplot
import matplotlib.pyplot as plt

# *********************************************************************************************
# Math Functions
# *********************************************************************************************

def powers_of_two (maximum, minimum=2):
    results = [minimum*minimum]
    power = 2
    while results[-1] < maximum/2:
        power += 1
        results.append (2**power)
    return results[:-1]

def vector_norm (vector):
    return sqrt(sum([x**2 for x in vector]))

def vector_diff (v1, v2):
    return list(map(operator.sub, v1, v2))

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
# Elbow Method For Selecting the Best K.
# *********************************************************************************************

# Input: Bics is a vector of vectors of the form [k bic].

# Methodology: Find the largest angle less 45 degrees between successive
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
# Kmeans Clustering
#*******************************************************************************************

def initial_centers (df, labels):
    centers = [list(df.loc[index]) for index in labels]
    return np.array(centers)


#--------------------------------------------------------------------

def compute_cluster_radius (cluster_df, cluster, center):
    members = cluster_df.loc[cluster_df['cluster'] == cluster]
    max_dist = 0
    for index, member in members.iterrows():
        point = (member['x'], member['y'])
        dist = distance.euclidean(center, point)
        if dist > max_dist:
            max_dist = dist
    return max_dist

#--------------------------------------------------------------------

def cluster_xy_data_kmeans (df, K=None, words=None):
    # Determine K if needed.
    if K==None:
        K, bics = cluster_xy_data_em (df)
    df = df.copy()
    # Convert to numpy array
    points = df.as_matrix()
    # A numpy array of shape (points, dimensions)
    #centers = initial_centers(df, labels)
    # Run Kmeans. Need to specify n_init to avoid warnings.
    km = KMeans(n_clusters=K, max_iter=10000).fit(points)
    df['cluster'] = list(km.labels_)
    return df, km.cluster_centers_, km

#--------------------------------------------------------------------

def compute_bics_for_k_range(df, kandidates=None):
    if kandidates==None:
        kandidates = powers_of_two(df.shape[0])
    # Generate a model for each k
    models = []
    for k in kandidates:
        ldf, centers, model = cluster_xy_data_kmeans (df, K=k)
        models.append([k, ldf, centers, model])
    # Find the best bic range. NB: Bic term has 3 values: bic, density, variance.
    # Use density value as BIC for now.
    stats = [[k, BIC(ldf, centers)] for k, ldf, centers, km in models]
    bics, dens, vars = extract_stats(stats)
    return bics, dens, vars

#--------------------------------------------------------------------

def find_best_k_range (df, kandidates=None, iterations=1000):
    if kandidates==None:
        kandidates = powers_of_two(df.shape[0])
    print ("Running K-Means " + str(len(kandidates)) + " times...")
    bics, dens, variance = compute_bics_for_k_range(df, kandidates)
    best_range = find_elbow (bics)
    return best_range

#--------------------------------------------------------------------
 
# Bayesian Based K Selector with logarithmic time search

def binsearch_bic_k (df, k_range, iterations=1000):
    r1, r2 = k_range
    k1, b1 = r1
    k2, b2 = r2
    print ("Running Kmeans " + str(int(np.log(k2 - k1))) + " times....")
                                       
    while True:
        if k2 <= k1:
            return [k1, b1]
        elif k2-k1==1:
            if b1<b2:
                return [k1, b1]
            else:
                return [k2, b2]
        else:
            mid_k = k1 + int(round((k2 - k1) / 2.0))
            print ("Next K:  " + str(mid_k))
            ldf, centers, km = cluster_xy_data_kmeans (df, K=mid_k)
            # Use density as BIC for now.
            bic = BIC(ldf, centers)
            mid_b = bic[2]
            if abs(b1 - mid_b) > abs(mid_b - b2):
                k1, b1 = mid_k, mid_b
            else:
                k2, b2 = mid_k, mid_b

#--------------------------------------------------------------------
# FIND BEST K
#--------------------------------------------------------------------

def find_best_k (df, kandidates=None, iterations=100):
    print ("Finding best K range...")
    k_range = find_best_k_range(df, kandidates=kandidates,iterations=iterations)
    print ("Best K Range: " + str(k_range))

    print ("Performing binary search on K range...")
    k, b = binsearch_bic_k(df, k_range, iterations=iterations)
    print ("Best K value is " + str(k))
           
    return k

    
#--------------------------------------------------------------------

def run_kmeans(df, K=None, iterations=1000):
    if K == None:
        K = find_best_k (df,  iterations=iterations)
    ldf, centers, km = cluster_xy_data_kmeans (df, K=K)
    return ldf, centers, K


#*******************************************************************************************
# Graphical Plots
#*******************************************************************************************

def add_circles (fig, centers, df):
    ax = fig.gca()
    for i, center in enumerate(centers):
        radius = compute_cluster_radius(df, i, center)
        circle = plt.Circle(center, radius, fill=False)
        #circle = plt.Circle(center, radius, color=i+1)
        ax.add_artist(circle)

#-------------------------------------------------------------------------------------------
# Plot Clusters
#-------------------------------------------------------------------------------------------

def show_styles():
    print (plt.style.available)

#-------------------------------------------------------------------------------------------

def set_axis_boundaries (df):
    xmin = min(df['x'])
    xmax = max(df['x']) + 5
    ymin = min(df['y'])
    ymax = max(df['y']) + 5
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    
#-------------------------------------------------------------------------------------------


def plot_clusters (df, centers, title="Clusters"):

    area = 4
    colors = df['cluster']
    fig = plt.figure()

    plot = fig.add_subplot(1, 1, 1)
    
    plot.scatter(df['x'], df['y'], s=area, c=colors)

    # Set the axes boudaries
    set_axis_boundaries(df)
    plt.style.use('seaborn-pastel')
    plt.title(title)
     
    x_centers = [p[0] for p in centers]
    y_centers = [p[1] for p in centers]
    plot.scatter(x_centers, y_centers, s=100, c=list(range(len(centers))))

    # Add cluster circles
    add_circles(fig, centers, df)
    
    # for  pos in centers:
    #     # word = cda.find_closest_word_to_point(df, pos)[0]
    #     words = cda.find_relevant_words_near_point(df, pos)
    #     index = list(df.index)
    #     for word in words:
    #         if word in index:
    #             print ("Word: " + word)
    #             pos = df.loc[word][['x', 'y']]
    #             plot.annotate(word, pos)

    plt.show()

#-------------------------------------------------------------------------------------------

# Select a k, run kmeans, plot clusters.

def plot_kmeans (df, K=None):
    ldf, centers, K = run_kmeans(df, K=K)
    title = ("Kmeans Clusters for K = " + str(K))
    plot_clusters(ldf, centers, title=title)
    return K, ldf, centers

#*******************************************************************************************
# Random Data Tests
#*******************************************************************************************

# ------------------------------------------------------------------------------------------
# Generates a grid of random real numbers.
# ------------------------------------------------------------------------------------------

def generate_random_grid (K, size, min=0, max=1000, margin=50):
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

def test_kmeans (K, data_size, margin=50):
    df = generate_random_grid(K, data_size, margin=margin)
    plot_kmeans(df, K)
    return True

# ------------------------------------------------------------------------------------------

def test_kselect (K, data_size, margin=50, plot=False):
    df = generate_random_grid(K, data_size, margin=margin)
    predicted_K = find_best_k (df)
    plot_kmeans(df, predicted_K)
    return True

# ------------------------------------------------------------------------------------------

def evaluate_kselect (number_of_ks=5, margin=50):
    K_range = map(lambda x: x*x, range(2, number_of_ks+2))
    data_sizes = [1000, 2000, 3000, 4000]
    results = []
    for K in K_range:
        for data_size in data_sizes:
            df = generate_random_grid(K, data_size, margin=margin)
            predicted_K = find_best_k (df)
            results.append([data_size, margin,K, predicted_K, abs(K-predicted_K)])
    columns = ['data_size', 'margin', 'correct_k', 'predicted_k', 'error']
    return(pd.DataFrame(results, columns=columns))
                
#*******************************************************************************************
# End
#*******************************************************************************************
