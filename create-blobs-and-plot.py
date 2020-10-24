# SRKR Internal Hackathon for SIH2020 Qualifiers.
# Model Design - Air Pollution Hotspot Detection

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import os
import re
import math
import geopandas as gpd
import copy

from datetime import date, time, datetime, timedelta
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray, gray2rgb

def read_data():
    files = os.listdir('weekly_bc_2015')
    files = [f for f in files if '.mat' in f]
    files = [f for f in files if 'weekly' in f]
    files = [(int(re.findall(r'\d+', f)[0]), f) for f in files]
    files = sorted(files, key=lambda x : x[0])
    files_copy = copy.deepcopy(files)
    files_copy[0] = (files[0][0], files[0][1], date(year=2015, month=9, day=23), date(year=2015, month=9, day=23) + timedelta(days=7))
    for i in range(1, len(files)):
        files_copy[i] = (files[i][0], files[i][1], files_copy[i-1][3], files_copy[i-1][3] + timedelta(days=7))
    return files_copy

def get_max_min_poll_values(files):
    all_poll_values = []
    for file in files:
        mat = scipy.io.loadmat('weekly_bc_2015/'+file[1])
        poll_arr_with_loc = mat['cc'].reshape((601, 601, 3))
        poll_arr = poll_arr_with_loc[:, :, 2]
        all_poll_values += poll_arr.tolist()

    max_pol = np.nanmean(all_poll_values)+3*np.nanstd(all_poll_values)
    min_pol = np.nanmin(all_poll_values)
    return max_pol, min_pol

def create_grayscale_image(file, max_pol, min_pol):
    mat = scipy.io.loadmat('weekly_bc_2015/'+file[1])
    poll_arr_with_loc = mat['cc'].reshape((601, 601, 3))
    poll_arr = poll_arr_with_loc[:, :, 2]
    poll_arr_copy = copy.deepcopy(poll_arr)
    poll_arr = (poll_arr - min_pol)/(max_pol - min_pol)
    poll_arr[poll_arr > 1] = 1
    for i in range(poll_arr.shape[0]):
        for j in range(poll_arr.shape[1]):
            if math.isnan(float(poll_arr[i, j])):
                poll_arr[i, j] = 0
    return poll_arr, poll_arr_copy

def create_blobs(image_gray):
    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.05)
    print(blob_log)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#     blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.15)
#     blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
#     print(blob_dog)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.005)
    print(blob_doh)

#     blobs_list = [blobs_log, blobs_dog, blobs_doh]
    blobs_list = [blobs_log, blobs_doh]
    return blobs_list
    
    

def plot_blobs(image_gray, poll_arr, blobs_list, lat_img_coordinates, long_img_coordinates, file, xmin, xmax, ymin, ymax):
#     colors = ['yellow', 'lime', 'red']
    colors = ['yellow', 'red']
#     titles = ['Original', 'Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
    titles = ['Original', 'Laplacian of Gaussian', 'Determinant of Hessian']
    sequence = zip(blobs_list, colors)
    dpi = 80
    height, width = image_gray.shape
    figsize = 3*(width / float(dpi)) + 1, 1*(height / float(dpi)) + 1

#     plt.figure(figsize=figsize)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ax = axes.ravel()
#     axes.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
#     plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    for idx in range(len(titles)):
        ax[idx].set_title(titles[idx])
        ax[idx].imshow(gray2rgb(image_gray))
        
        for coord_idx, (lat, long) in enumerate(zip(lat_img_coordinates, long_img_coordinates)):
            c = plt.Circle((long, lat), 1, color='blue', linewidth=1, fill=False)
            ax[idx].add_patch(c)
         
    df_blobs_data = []
    for idx, (blobs, color) in enumerate(sequence):
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx+1].add_patch(c)
#             print(x, ', ', y, ', ', r)
            if idx == 0:
                df_blobs_data.append([xmin+(x/100.0), ymax-(y/100.0), r/100.0, np.nanmean(poll_arr[int(max(0, x-math.floor(r))):int(min(image_gray.shape[0], x+math.floor(r))), int(max(0, y-math.floor(r))):int(min(image_gray.shape[0], y+math.floor(r))) ]), 'log'])
            else:
                df_blobs_data.append([xmin+(x/100.0), ymax-(y/100.0), r/100.0, np.nanmean(poll_arr[int(max(0, x-math.floor(r))):int(min(image_gray.shape[0], x+math.floor(r))), int(max(0, y-math.floor(r))):int(min(image_gray.shape[0], y+math.floor(r))) ]), 'doh'])
#         ax[idx+1].set_axis_off()
    df_blobs = pd.DataFrame(data=df_blobs_data, columns=['x', 'y', 'r', 'avg_pm2.5_value', 'type'])
    df_blobs.sort_values('avg_pm2.5_value', inplace=True, ascending=False)
    

    plt.tight_layout()
    date_component_name = str(file[2].day)+'-'+str(file[2].month)+'_'+str(file[3].day)+'-'+str(file[3].month)
    df_blobs.to_csv('blobs/'+file[1].split('.')[0]+'_'+date_component_name+'.csv')
    plt.savefig('images/'+file[1].split('.')[0]+'_'+date_component_name+'.png')
    plt.show()
    
# def superimpose_delhi_map():

def get_lat_long_image_coordinates(xmin, ymin):
    df = gpd.read_file('Municipal_Spatial_Data/Delhi/Delhi_Boundary.geojson')
    long_array = list(df.loc[0, 'geometry'].exterior.coords.xy[0])
    long_array = list(map(lambda x : round(x, 2), long_array))
    lat_array = list(df.loc[0, 'geometry'].exterior.coords.xy[1])
    lat_array = list(map(lambda x : round(x, 2), lat_array))

    long_img_coordinates = list(map(lambda x : int(100*round(x - xmin, 2)), long_array))
    lat_img_coordinates = list(map(lambda x : int(100*round(ymax - x, 2)), lat_array))
    return lat_img_coordinates, long_img_coordinates
    

files = read_data()
ymin = np.min(scipy.io.loadmat('weekly_bc_2015/'+files[0][1])['cc'][:, 0])
ymax = np.max(scipy.io.loadmat('weekly_bc_2015/'+files[0][1])['cc'][:, 0])
xmin = np.min(scipy.io.loadmat('weekly_bc_2015/'+files[0][1])['cc'][:, 1])
xmax = np.max(scipy.io.loadmat('weekly_bc_2015/'+files[0][1])['cc'][:, 1])

lat_img_coordinates, long_img_coordinates = get_lat_long_image_coordinates(xmin, ymin)

max_pol, min_pol = get_max_min_poll_values(files)
for file in files:
    # image_gray = rgb2gray(poll_arr)
    image_gray, poll_arr = create_grayscale_image(file, max_pol, min_pol)
    print(image_gray.shape)
    blobs_list = create_blobs(image_gray)
    plot_blobs(image_gray, poll_arr, blobs_list, lat_img_coordinates, long_img_coordinates, file, xmin, xmax, ymin, ymax)
    plt.imshow(gray2rgb(image_gray))
