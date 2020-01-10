import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from PIL import Image

#defining constants RGB
RED = 0
GREEN = 1
BLUE = 2


def get_clusters(source_img):

    #reading source image
    source = cv2.imread(source_img)
    #convert from BGR to RGB
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    #reshape to list of pixels
    new_source = source.reshape((source.shape[0] * source.shape[1], 3))
    print(source.shape)

    #cluster pixels
    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(new_source)

    #get dominant colors
    clusters = kmeans.cluster_centers_
    #converting to int
    clusters = clusters.astype(int)

    return clusters

def get_color_map(source, dest):
    
    #initiating dictionary to hold the color mappings
    color_map = {}

    index = 0

    while index < len(source):

        #getting the rgb mappings
        r = source[index][RED] - dest[index][RED]
        g = source[index][GREEN] - dest[index][GREEN]
        b  = source[index][BLUE] - dest[index][BLUE]
        
        #add values to dictionary
        color_map[index] = [r, g, b]
        print(r, g, b)
        index += 1

    print(color_map) 

def change_colors(color_map, dest_img):

    #reading source image
    img = cv2.imread(dest_img)
    #convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #reshape to list of pixels
    new_img = img.reshape((img.shape[0] * img.shape[1], 3))

    #cluster pixels
    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(new_img)

    #get labels
    labels = kmeans.labels_
    
    #declaring new dimensions
    x = int(labels.shape[0]/img.shape[1])
    y = int(labels.shape[0]/img.shape[0])
    #reshape labels
    labels = labels.reshape(x, y, 1)
    print(labels.shape)
    print(img.shape)

    #iterate through img updating pixel colors
    

    #img.save('newimage.png')

    #img.show()

#get the fire clusters and print
fire = get_clusters('poke/fire/charmander.png')
print("fire:")
print(fire)

#get the water clusters and print
water = get_clusters('poke/water/vaporeon.png')
print("water:")
print(water)

#create the color map using the clusters
color_map = get_color_map(fire, water)

change_colors(color_map, 'poke/water/vaporeon.png')



