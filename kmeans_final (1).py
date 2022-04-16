import numpy as np
import seaborn

np.random.seed(555)  #set the seed
K = 2 #no of clusters
max_iter = 50  #max iterations to run the program
X = np.array([[2, 4],[1.7, 2.8],[7, 8],[8.6, 8],[3.4, 1.5],[9,11]])
no_example = X.shape[0]
no_feat = X.shape[1]

def random_centroids_initialization(X):
    """
    initilizes the random points as centriods
    """
    c = np.zeros((K, no_feat))
    for k in range(K):
        centroid = X[np.random.choice(range(no_example))]
        c[k] = centroid
    return c

def create_clusters(X, c):
    """
    create a cluster to the nearest point based on eculdian distance
    """
    clusters = [[] for t in range(K)]

    for point_id, point in enumerate(X):
        nearest_centroid = np.argmin(
            np.sqrt(np.sum((point - c) ** 2, axis=1))
        )
        clusters[nearest_centroid].append(point_id)

    return clusters

def new_centroids(clusters, X):
    """
    create a new cluster based on mean
    """
    cs = np.zeros((K, no_feat))
    for i, c in enumerate(clusters):
        n_c = np.mean(X[c], axis=0)
        cs[i] = n_c

    return cs

def cluster_prediction(clusters, X):
    """
    predict the cluster
    """
    y_pred = np.zeros(no_example)
    for c_id, c in enumerate(clusters):
        for s_id in c:
            y_pred[s_id] = c_id

    return y_pred

def plot_fig(X, y):
    """
    plot the graph to display the clusters
    """
    seaborn.scatterplot(X[:, 0], X[:, 1],c=y)   
    
def custm_fit(X):
    """
    fit the kmeans cluster alogritham for a given data points
    """
    cen = random_centroids_initialization(X)
    for it in range(max_iter):
        clusters = create_clusters(X, cen)
        p_c = cen
        c = new_centroids(clusters, X)
        d = c - p_c
        if not d.any():
            print("Ended")
            break
    y_pred = cluster_prediction(clusters, X)
    plot_fig(X, y_pred)
    return y_pred

y_pred = custm_fit(X)