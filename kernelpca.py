import os
import numpy as np
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from matplotlib.colors import to_rgba
from functions_EDX import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler, StandardScaler
from skimage import exposure
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import rank
from skimage.morphology import disk
from skimage.morphology import black_tophat, disk, white_tophat
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.kernel_approximation import Nystroem
import joblib
# Define the directory containing the .npy files
directory = '/scratch/p301644/spectra/binned/FilteredImages/'

# Initialize an empty list to store the arrays
array_list = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        # Load the .npy file
        file_path = os.path.join(directory, filename)
        data = StandardScaler(with_std=False).fit_transform(np.load(file_path))
        
        # Append the data to the list
        array_list.append(data)

# Concatenate all arrays into a single array
concatenated_array = np.concatenate(array_list, axis=0)

pca= PCA(.95)
random_samples =np.random.choice(30*1024*1024,10000)

feature_map_nystroem = Nystroem(kernel='rbf', n_components=1000, random_state=1000, gamma=0.1)
transformed_data=[]
feature_map_nystroem.fit(concatenated_array[random_samples,:])

reshape_concatenated_array= concatenated_array.reshape(30,1024*1024,-1)

for i in range(30):
  transformed_data= feature_map_nystroem.transform(reshape_concatenated_array[i])
  
  np.save(os.path.join(directory,f'kpca_1_transfomed_data{i}.npy'), transformed_data)

    


