import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# checking existance of dir for saving extracted sift featuers
REL_SAVING_PATH = '../data/descriptors/sift'
SIFT_DESCRIPTOR_DIR = os.path.join(os.getcwd(), REL_SAVING_PATH)
os.makedirs(SIFT_DESCRIPTOR_DIR, exist_ok=True) # Ensure directory exists

# extracting sift features for given filepaths.txt
def extract_sift_features(img_list_file_path, imgs_root_path):
    # creating sift
    sift = cv.SIFT_create()

    # getting list of image paths
    with open(img_list_file_path, 'r') as f:
        img_paths = [path.strip() for path in f if path.strip()]

    # looping over paths
    for img_path in img_paths:
        # read only of img-like extension files
        if img_path.endswith(('jpg', 'png', 'jpeg')):
            img_grayscaled = cv.imread(os.path.join(imgs_root_path, img_path), cv.IMREAD_GRAYSCALE)
        else:
            continue
        
        # check if cv2 could read img
        if img_grayscaled is None: 
            print(f'[Error] Cv2 couldn\'t read image at ./{img_path}')
            continue

        # resize image to aviod exposion of my pc
        img_grayscaled = cv.resize(img_grayscaled, (512, 512))

        # get descriptors (2D matrix - each row is 128 describing property)
        _, descriptors = sift.detectAndCompute(img_grayscaled, None)

        # check for no descriptor error 
        if(descriptors.shape[0] == 0): 
           print('[Error] No Descriptors found at ./{img_path}!')

        # getting filename 
        npy_filename = os.path.basename(img_path.rsplit('.', 1)[0] + '.npy')
        
        # save descriptors in REL_SAVING_PATH as filename.npy
        np.save(os.path.join(REL_SAVING_PATH, npy_filename) , descriptors)