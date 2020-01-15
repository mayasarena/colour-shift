import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
import sys

#defining constants HSV
HUE = 0
SATURATION = 1
VALUE = 2

CLUSTER_NUM = 3

def get_clusters(source_img):

    #reading source image
    source = cv2.imread(source_img)
    #convert from BGR to HSV
    source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    #reshape to list of pixels
    new_source = source.reshape((source.shape[0] * source.shape[1], 3))

    #cluster pixels
    kmeans = KMeans(n_clusters = CLUSTER_NUM, init = 'k-means++', max_iter = 100, n_init = 10, verbose = 0, random_state = 1000)
    kmeans.fit(new_source)

    #get dominant colors
    clusters = kmeans.cluster_centers_
    labels = kmeans.labels_
    #converting to int
    clusters = clusters.astype(int)

    #plot for debugging
    #plt.scatter(new_source[:, 0], new_source[:, 1], c=kmeans.labels_, cmap='hsv')
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], color='black')
    #plt.show()

    return clusters, labels

def order(source_labels, dest_labels):

    #initiating an empty dicttionary to hold number matchups
    s_order = {}
    d_order = {}

    index = 0
    count = 0

    #getting the number of pixels belonging to each cluster
    s_unique, s_count = np.unique(source_labels, return_counts = True)
    #copying array and sorting it 
    s_sorted = s_count.copy()
    s_sorted.sort()
    s_sorted = s_sorted[::-1]
    
    d_unique, d_count = np.unique(dest_labels, return_counts = True)
    d_sorted = d_count.copy()
    d_sorted.sort()
    d_sorted = d_sorted[::-1]

    #matching up the number of pixels in each cluster to index and storing result
    #in a dictionary
    for index in range (len(s_unique)):
        curr_number = s_count[index]
        for count in range(len(s_unique)):
            number = s_sorted[count]
            if curr_number == number:
                s_order[count] = index
    
    for index in range (len(d_unique)):
        curr_number = d_count[index]
        for count in range(len(d_unique)):
            number = d_sorted[count]
            if curr_number == number:
                d_order[count] = index

    return s_order, d_order
    
def get_color_map(source, dest, s_order, d_order):
    
    #initiating dictionary to hold the color mappings
    color_map = {}

    index = 0

    print(s_order[index])
    print(d_order[index])

    while index < len(source):

        #getting the hsv mappings
        h = source[s_order[index]][HUE]
        s = source[s_order[index]][SATURATION] - dest[d_order[index]][SATURATION]
        v = source[s_order[index]][VALUE] - dest[d_order[index]][VALUE]
        
        #add values to dictionary
        color_map[index] = [h, s, v]
        index += 1

    return color_map

def change_colors(color_map, dest_img, labels, d_order):

    #reading source image
    img = cv2.imread(dest_img)
    #convert from BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #reshape to list of pixels
    new_img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    #declaring new dimensions
    x = int(labels.shape[0]/img.shape[1])
    y = int(labels.shape[0]/img.shape[0])
    #reshape labels
    labels = labels.reshape(x, y)

    #iterate through img updating pixel colors
    x = 0
    y = 0
    
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            #get current rgb values
            hsv = img[x][y]
            label = labels[x][y]
            #get color mapping
            mapping = color_map[label]
           
            #detect white/grey
            if 0 <= hsv[SATURATION] <= 5 and 128 <= hsv[VALUE] <= 255:
                h = mapping[HUE]
                s = hsv[SATURATION]
                v = hsv[VALUE] 

            #detect black/grey
            elif 0 <= hsv[SATURATION] <= 175 and 0 <= hsv[VALUE] <= 127:
                h = mapping[HUE] 
                s = hsv[SATURATION] 
                v = hsv[VALUE]

            #calculate new hsv values
            else:  
                h = mapping[HUE]
                s = hsv[SATURATION] + mapping[SATURATION] 
                v = hsv[VALUE] + mapping[VALUE]

            #check that they are all in bounds
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

            #set new hsv values
            img[x][y][HUE] = h
            img[x][y][SATURATION] = s
            img[x][y][VALUE] = v

    #print img pixels into file

    np.set_printoptions(threshold=np.inf, linewidth=img.shape[1])
    with open('output.txt', 'a') as f:
        print(img, file = f)
    

    #save img, convert back to rgb, and display
    final = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite('newimage.png', final)
    cv2.imshow('image', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

source = 'source.png'
dest = 'dest.png'
#get the fire clusters and print
source_clusters, source_labels = get_clusters(source)

#get the water clusters and print
dest_clusters, dest_labels = get_clusters(dest)

s_order, d_order = order(source_labels, dest_labels)

#create the color map using the clusters
color_map = get_color_map(source_clusters, dest_clusters, s_order, d_order)

print(source_clusters)
print(dest_clusters)
print(color_map)

change_colors(color_map, dest, dest_labels, d_order)



