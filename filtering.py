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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from skimage import exposure
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import rank
from skimage.morphology import disk
from skimage.morphology import black_tophat, disk, white_tophat
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.kernel_approximation import Nystroem
import matplotlib.cm as cm

from PIL import Image
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.kernel_approximation import Nystroem



def filter(image, k,s):
    min_value = np.min(image)
    max_value = np.max(image)

# Invert the array
    image = min_value + max_value - image
    local_mean = uniform_filter(image, size=k)
    local_var = uniform_filter(image**2, size=k) - local_mean**2
    local_std = np.sqrt(local_var)
    def local_gaussian_filter(image, local_mean, local_std):
        local_std = np.where(local_std > 0, local_std, 1)
        normalized_image = (image - local_mean) / local_std
        filtered_image = gaussian_filter(normalized_image, sigma=s)
        return (filtered_image * local_std + local_mean)

    filtered_image= local_gaussian_filter(image, local_mean, local_std)

    return filtered_image

def process_channel(image, sigma):
    image = filter(image, k=(4*sigma)+1, s=sigma)
    mp = image
    return mp

def filtering(base_image, sigma):
    n_images, base_image_rows, base_image_columns, base_image_channels = base_image.shape
    emp = np.zeros((n_images, base_image_rows, base_image_columns, base_image_channels))
    for n in range(n_images):

        results =(process_channel(base_image[n, :, :, channel], sigma) for channel in range(base_image_channels))

        for channel, result in enumerate(results):
            emp[n, :, :, channel] = result

    return emp



feature_map_nystroem = Nystroem(kernel='rbf', n_components=250, random_state=1000, gamma=None)
transformed_data=[]
selected_samples=[]
# Mount Google Drive to access and save files


# Define the Google Drive path where you want to save the processed data
save_folder = r'/scratch/p301644/spectra/binned/Clusteringfilter/'
spectrum_folder = r'/scratch/p301644/spectra/binned/'

scaler= StandardScaler()
file_names = []
file_names1 = []
images =[]
nch=4000
global_sum = np.zeros((nch,))
global_sum_sq = np.zeros((nch,))

tmp = os.listdir(spectrum_folder)
for filename in tmp:
    if filename.endswith('.npz'):
        file_names.append(os.path.join(spectrum_folder,filename))

file_names = sorted(file_names)
print(file_names)
tilenos= np.arange(0,30)
X_ZCA_rescaled=[]
nch=250
s=1024
s1=1024
n=1
en=len(tilenos)
gray_images=[]
images1=[]
# tile10=r"C:\Users\P301644\Documents\Data\Dataset2\haadf_png\10.png"
# image1 = np.array(Image.open(tile10))

n_filters=2
filter_size1= 1
standardized_images_path = os.path.join(spectrum_folder, 'FilteredImages')
os.makedirs(standardized_images_path, exist_ok=True)   
filter_size2=5
for tile_idx in tilenos:
    print(file_names[tile_idx])
    spectrum_2D = np.load(file_names[tile_idx], allow_pickle=False)['spectrum_2D']
    spectrum_2D_n = np.reshape(spectrum_2D, (1024, 1024, nch))
    spectrum_2D_n1 = spectrum_2D_n[ 0:, 0:,:nch]

    norm_flat_spectrum_2D_n = np.reshape(spectrum_2D_n1, (s * s, nch))
    data=norm_flat_spectrum_2D_n.reshape(-1,nch)
    data_for_filtering =data.reshape(1, s, s, nch)[:,:,:,:]

    filtered_channels= filtering(data_for_filtering,filter_size1)

    filtered_channels2= filtering(data_for_filtering, filter_size2)

    spectrum_2D =np.concatenate((filtered_channels2.reshape(-1,nch), filtered_channels.reshape(-1,nch)), axis=-1)

    Y =np.asarray(spectrum_2D, dtype=np.float32)
    standardized_images_path1 = os.path.join(standardized_images_path, f'{tile_idx}_filtered')
    np.save(standardized_images_path1, Y)
    





