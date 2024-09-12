# imports

import numpy as np
from sklearn.decomposition import PCA
# import umap
from functions_EDX import *
import time
import matplotlib.pyplot as plt
from scipy.stats import zscore
from datetime import datetime
from skimage.feature import peak_local_max
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
import os
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('file_numbers',nargs='+',type=str)
#args = parser.parse_args()


home_path = r'C:\Users\P301644\Documents\Data\Dataset2\EDX\Raw input files'

#file_numbers = args.file_numbers #['0142 HAADF 12000 x_20','0202 HAADF 12000 x_20','0408 HAADF 12000 x_20','0428 HAADF 12000 x_20','0634 HAADF 12000 x_20','0654 HAADF 12000 x_20']
#file_names = ['%04d HAADF 12000 x_20' % int(i) for i in file_numbers]



# Get all file names (no extension)
file_names = []
for file in os.listdir(home_path):
    if file.endswith('.npz'):
        file_names.append(file[:-4])



exp_tag = 'Binned'
for i in range(1):
    # load file
    file_name = file_names[i]
    time_str = str(datetime.now())[:16]

    loaded_file = np.load(os.path.join(home_path,file_name+'.npz'))
    haadf = loaded_file['haadf']
    spectrum = loaded_file['spectrum'][:,:,96:]
    xray_energies = loaded_file['xray_energies'][96:]
    subsample_size = spectrum.shape[2]


    # Clean up then bin the spectrum and check if any empty channels remain
    n_bins = 4000
    # spectrum = rebin_spectrum(spectrum,n_bins)

    # Now bin in XY
    subsample_size = 128
    spectrum = rebin_spectrumXY(spectrum.reshape(2048,2048,4000),subsample_size)  
    haadf = rebin_XY(haadf,subsample_size)       

    # normalize each pixel over peak (idx = 3)
    #spectrum = np.array([j/j[3] for i in spectrum for j in i]).reshape((1024,1024,250)) 

    xray_energies = rebin_energies(xray_energies,n_bins)
    where_notempty = ~np.all(spectrum==0,axis=(0,1))
    spectrum = spectrum[:,:,where_notempty]
    spectral_depth = spectrum.shape[2]
    spectrum_2D = np.reshape(spectrum,(subsample_size*subsample_size,spectral_depth))
    print("%04d channels remain" % spectral_depth)
    np.savez_compressed(os.path.join(home_path,'Binned4000','_%s_%s.npz' % (file_name,exp_tag)), spectrum_2D=spectrum_2D)