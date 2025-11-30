import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # avoiding explosion of my pc


# loading descriptors from /descriptor_dir/{f} for f in image_list_file
def load_descriptors(descriptor_dir, image_list_file):
    with open(image_list_file, 'r') as f:
        img_paths = [p.strip() for p in f if p.strip()]
    
    all_desc = []
    for img_path in img_paths:
        fname = os.path.basename(img_path).rsplit('.', 1)[0] + '.npy'
        file_path = os.path.join(descriptor_dir, fname)
        if os.path.exists(file_path):
            desc = np.load(file_path)
            if desc is not None and desc.shape[0] > 0:
                all_desc.append(desc)
    all_desc = np.vstack(all_desc)
    
    return all_desc


# kmeans clusterisation with sklearn.MiniBatchKMeans
def fit_kmeans(descriptors, k):
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, verbose=1, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

# make higtogram for single image's descriptors 
def build_histogram(descriptors, kmeans):
    if descriptors is None or descriptors.shape[0] == 0:
        return np.zeros(kmeans.n_clusters)
    
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(float)
    hist /= hist.sum()  # normalize
    return hist

