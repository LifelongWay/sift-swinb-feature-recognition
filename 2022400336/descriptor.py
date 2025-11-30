import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# for swinb
from PIL import Image
import torch
from torchvision.models import swin_b, Swin_B_Weights



##      -- sift --       ##

# checking existance of dir for saving extracted sift featuers
SIFT_REL_SAVING_PATH = '../data/descriptors/sift'
SIFT_DESCRIPTOR_DIR = os.path.join(os.getcwd(), SIFT_REL_SAVING_PATH)
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
        
        # save descriptors in sift_REL_SAVING_PATH as filename.npy
        np.save(os.path.join(SIFT_REL_SAVING_PATH, npy_filename) , descriptors)



##      -- swin_b --       ##
SWIN_REL_SAVING_PATH = '../data/descriptors/swin'
SWIN_DESCRIPTOR_DIR = os.path.join(os.getcwd(), SWIN_REL_SAVING_PATH)
os.makedirs(SIFT_DESCRIPTOR_DIR, exist_ok=True) # Ensure directory exists


def extract_swin_b_features(img_list_file_path, imgs_root_path):
    # loading pretrained weights from IMAGENET1K_V1
    weights = Swin_B_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # loading feature extractor
    model = swin_b(weights = weights)
    model = model.features
    model.eval()

    # loop over image paths
    with open(img_list_file_path, 'r') as f:
        img_paths = [p.strip() for p in f if p.strip()]
        
    
    for img_path in img_paths:

        # reading only with img-extensions
        if not img_path.endswith(('jpg', 'jpeg', 'png')):
            continue

        full_path = os.path.join(imgs_root_path, img_path)

        # try loading image
        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            print(f"[Error] PIL couldn't read image at {full_path}")
            continue

        # apply pretrained transforms
        img_tensor = preprocess(img).unsqueeze(0)  # shape (1, 3, 224, 224)

        # forward pass
        with torch.no_grad():
            feats = model(img_tensor)        # (1, 1024, 7, 7)

        # convert to shape (49, 1024)
        feats = feats.squeeze(0)             # (1024, 7, 7)
        feats = feats.flatten(1).T           # (49, 1024)
        descriptors = feats.cpu().numpy()

        if descriptors.shape[0] == 0:
            print(f'[Error] No descriptors found at {img_path}!')
            continue

        # build output filename (replace .jpg → .npy)
        npy_filename = os.path.basename(img_path.rsplit('.', 1)[0] + '.npy')

        # save to directory
        np.save(os.path.join(SWIN_REL_SAVING_PATH, npy_filename), descriptors)

        print(f"[OK] Saved {descriptors.shape} descriptors for {img_path}")