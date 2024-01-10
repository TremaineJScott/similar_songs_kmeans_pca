
from sklearn.cluster import KMeans

def apply_kmeans(data, n_clusters=5):

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    
    return kmeans, labels
