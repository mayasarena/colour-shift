# Author: Maya Murad
# Color-shift, shifts the colors from one image to another

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
import sys

# Defining constants HSV
HUE = 0
SATURATION = 1
VALUE = 2

# Defining the number of clusters
CLUSTER_NUM = 5

# Method get_clusters uses KMeans to get clusters of colors from an image
# Parameter:
# source_img: the image name (String)
def get_clusters(source_img):

    # Reading source image
    source = cv2.imread(source_img)
    # Convert from BGR to HSV
    source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    # Reshape to list of pixels
    new_source = source.reshape((source.shape[0] * source.shape[1], 3))

    # Cluster pixels
    kmeans = KMeans(n_clusters = CLUSTER_NUM, init = 'k-means++', max_iter = 100, n_init = 10, verbose = 0, random_state = 1000)
    kmeans.fit(new_source)

    # Get dominant colors
    clusters = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Cconverting to int
    clusters = clusters.astype(int)

    #plot for debugging
    #plt.scatter(new_source[:, 0], new_source[:, 1], c=kmeans.labels_, cmap='hsv')
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], color='black')
    #plt.show()

    return clusters, labels

# Method order takes the clusters and matches them up according
# to the number of labels in each cluster, therefore the most
# dominant clusters of each image will be associated with each 
# other since KMeans does not order clusters by dominance
# Parameters:
# source_labels, dest_labels: the labels from each image, received
# from KMeans
def order(source_labels, dest_labels):

    # Initializing an empty dicttionary to hold number matchups
    s_order = {}
    d_order = {}

    index = 0
    count = 0

    # Getting the number of pixels belonging to each cluster
    s_unique, s_count = np.unique(source_labels, return_counts = True)
    
    # Copying the array and sorting it 
    s_sorted = s_count.copy()
    s_sorted.sort()
    s_sorted = s_sorted[::-1]
    
    # Same thing as above for the other img
    d_unique, d_count = np.unique(dest_labels, return_counts = True)
    d_sorted = d_count.copy()
    d_sorted.sort()
    d_sorted = d_sorted[::-1]

    # Matching up the number of pixels in each cluster to index and storing result
    # in a dictionary
    for index in range (len(s_unique)):
        curr_number = s_count[index]
        for count in range(len(s_unique)):
            number = s_sorted[count]
            if curr_number == number:
                s_order[count] = index
    
    # Same thing but for other img
    for index in range (len(d_unique)):
        curr_number = d_count[index]
        for count in range(len(d_unique)):
            number = d_sorted[count]
            if curr_number == number:
                d_order[count] = index

    return s_order, d_order

# Method get_color_map creates a dictionary that holds
# the color changes to be made for each cluster
# Parameters:
# source, dest: the clusters from each img
# s_order, d_order: dictionaries from the order method
def get_color_map(source, dest, s_order, d_order):
    
    # Initializing a dictionary to hold the color mappings
    color_map = {}

    index = 0

    while index < len(source):

        # Calculating the hsv mappings
        h = source[s_order[index]][HUE]
        s = source[s_order[index]][SATURATION] - dest[d_order[index]][SATURATION]
        v = source[s_order[index]][VALUE] - dest[d_order[index]][VALUE]
        
        # Add the values to the dictionary
        color_map[index] = [h, s, v]
        index += 1

    return color_map

# Method change_colors goes through the destination image pixel by
# pixel and changes the colors
# Paramaters:
# color_map: the dictionary returned from the get_color_map method
# dest_img: the destination image (String)
# labels: the destination labels, received from KMeans
# d_order: the destination order, received from the order method
def change_colors(color_map, dest_img, labels, d_order):

    # Reading source image
    img = cv2.imread(dest_img)
    # Convert from BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Reshape to list of pixels
    new_img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    # Declaring new dimensions of the array of pixels
    x = int(labels.shape[0]/img.shape[1])
    y = int(labels.shape[0]/img.shape[0])
    # Reshape the labels
    labels = labels.reshape(x, y)

    x = 0
    y = 0
    
    # Iterate through img updating pixel colors
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # Get current hsv values
            hsv = img[x][y]
            label = labels[x][y]
            # Get color mapping
            mapping = color_map[label]
           
            # Detect white/grey
            if 0 <= hsv[SATURATION] <= 5 and 128 <= hsv[VALUE] <= 255:
                h = mapping[HUE]
                s = hsv[SATURATION]
                v = hsv[VALUE] 

            # Detect black/grey
            elif 0 <= hsv[SATURATION] <= 175 and 0 <= hsv[VALUE] <= 127:
                h = mapping[HUE] 
                s = hsv[SATURATION] 
                v = hsv[VALUE]

            # Calculate new hsv values
            else:  
                h = mapping[HUE]
                s = hsv[SATURATION] + mapping[SATURATION] 
                v = hsv[VALUE] + mapping[VALUE]

            # Check that they are all in bounds
            if h < 0:
                h = 0
            if s < 0:
                s = 0
            if v < 0:
                v = 0
            if h > 180:
                h = 180
            if s > 255:
                s = 255
            if v > 255:
                v = 255

            # Cet new hsv values
            img[x][y][HUE] = h
            img[x][y][SATURATION] = s
            img[x][y][VALUE] = v

    # Print the img pixels into file

    np.set_printoptions(threshold=np.inf, linewidth=img.shape[1])
    with open('output.txt', 'a') as f:
        print(img, file = f)
    

    # Save img, convert back to rgb, and display
    final = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite('newimage.png', final)
    cv2.imshow('image', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# The source and destination images
source = 'source.jpg'
dest = 'dest.jpg'

# Get the source clusters and labels
source_clusters, source_labels = get_clusters(source)

# Get the destination clusters and labels
dest_clusters, dest_labels = get_clusters(dest)

# Get the source and destionation orders
s_order, d_order = order(source_labels, dest_labels)

# Create the color map using the clusters
color_map = get_color_map(source_clusters, dest_clusters, s_order, d_order)

# Change the colors
change_colors(color_map, dest, dest_labels, d_order)



