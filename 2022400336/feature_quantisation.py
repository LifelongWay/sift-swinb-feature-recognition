import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # avoiding explosion of my pc


# loading descriptors from /descriptor_dir/{f} for f in image_list_file
def load_descriptors(descriptor_dir, image_list_file):
    with open(image_list_file, 'r') as f:
        img_paths = [p.strip() for p in f if p.strip()]
    
    all_desc = []
    n = 1
    for img_path in img_paths:
        print('iter ', n)
        if n == 5300: break # hits my limit at 5360
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

def build_histograms_for_list(image_list_file, descriptor_dir, hist_save_dir, kmeans, k):
    os.makedirs(hist_save_dir, exist_ok=True)

    with open(image_list_file, "r") as f:
        img_paths = [p.strip() for p in f if p.strip()]

    for img_path in img_paths:
        base = os.path.basename(img_path).rsplit('.', 1)[0]
        
        desc_path = os.path.join(descriptor_dir, base + ".npy")
        if not os.path.exists(desc_path):
            print(f"[Warning] Descriptor missing for {img_path}")
            continue

        desc = np.load(desc_path)
        hist = build_histogram(desc, kmeans)

        out_path = os.path.join(hist_save_dir, f"{base}_k{k}.npy")
        np.save(out_path, hist)


# final pipeline
def run_feature_quantization(
    train_list,
    test_list,
    sift_desc_dir,
    swin_desc_dir,
    sift_hist_dir,
    swin_hist_dir,
    ks=[50, 100, 500]
):
    for k in ks:
        print(f"\n[INFO] Feature quantisation for K = {k}.... ")

        # --- SIFT ---
        if sift_desc_dir and sift_hist_dir:
            # loading extracted features
            sift_train_desc = load_descriptors(sift_desc_dir, train_list)
            # quantising extracted features
            sift_kmeans = fit_kmeans(sift_train_desc, k)
            # building histograms
            build_histograms_for_list(train_list, sift_desc_dir, sift_hist_dir, sift_kmeans, k)
            build_histograms_for_list(test_list,  sift_desc_dir, sift_hist_dir, sift_kmeans, k)
            print("SIFT DESCRIPTORS ARE DONE !")
        # --- Swin-B ---
        if swin_desc_dir and swin_hist_dir:
            # loading extracted features
            swin_train_desc = load_descriptors(swin_desc_dir, train_list)
            # quantising extracted features
            swin_kmeans = fit_kmeans(swin_train_desc, k)
            # building histograms
            build_histograms_for_list(train_list, swin_desc_dir, swin_hist_dir, swin_kmeans, k)
            build_histograms_for_list(test_list,  swin_desc_dir, swin_hist_dir, swin_kmeans, k)

    print("\n[INFO] Feature quantization completed.")