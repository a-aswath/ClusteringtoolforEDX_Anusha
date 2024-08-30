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
import matplotlib.cm as cm
# Define the directory containing the .npy files
directory = '/scratch/p301644/spectra/binned/FilteredImages/'

from sklearn.cluster import MiniBatchKMeans
# Initialize an empty list to store the arrays
array_list = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.startswith("kpca"):
        # Load the .npy file
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)
        
        # Append the data to the list
        array_list.append(data)

# Concatenate all arrays into a single array
concatenated_array = np.concatenate(array_list, axis=0)
mbkmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000, init= 'k-means++', random_state=1)
pca= PCA(.95)
random_samples =np.random.choice(30*1024*1024,10000)
transformed_data=[]

reshape_concatenated_array= concatenated_array.reshape(30,1024*1024,-1)
std_path= os.path.join(directory,'ClusteringResults10')
os.makedirs(std_path, exist_ok=True)  
s=1024
no_of_clusters=10
for i in range(30):
  mbkmeans.partial_fit(reshape_concatenated_array[i])
for i in range(30):
  labels= mbkmeans.predict(reshape_concatenated_array[i])
  all_distances= mbkmeans.transform(reshape_concatenated_array[i]).reshape(1,-1)
  normalized_distances =(1- (all_distances - np.min(all_distances)) / (np.max(all_distances) - np.min(all_distances))).reshape(1,s*s,no_of_clusters)
  for j in range(no_of_clusters):

 
        relevant_image= (normalized_distances[0,:,j]* (labels==j)).reshape(1024,1024)
        r = (255 * (relevant_image - np.min(relevant_image)) / (np.max(relevant_image) - np.min(relevant_image))).astype(np.uint8)
            
        # Reshape to the desired shape
        r_reshaped = r.reshape(s, s)
            
        # Apply the 'hot' colormap
        colormap = cm.get_cmap('gray')
        colored_image = colormap(r_reshaped / 255.0)  # Normalize to [0, 1] for colormap
            
            # Convert the colored image to uint8 format
        colored_image_uint8 = (colored_image[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(colored_image_uint8)
        output_path = f'{i}_{j}.png'
        image.save(os.path.join(std_path,output_path) ) 
  


    
