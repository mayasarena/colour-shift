import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from PIL import Image


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



def change_colors(color_map, dest_img):

    #load image
    img = Image.open(dest_img)

    #convert to RGB
    img = img.convert('RGB')
    
    #load the pixels
    pixels = img.load()
    print(img.size)
    #print(list(img.getdata()))

    color = 0
    #print('color:', dest[color])
    #change pixel colors
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            #print('pixels: ', pixels[i,j])
            if pixels[i,j] == (dest[color][0], dest[color][1], dest[color][2]):
                pixels[i,j] = (source[color][0], source[color][1], source[color][2])
                color = color + 1

    img.save('newimage.png')

    img.show()

#get the fire clusters and print
fire = get_clusters('poke/fire/charmander.png')
print("fire:")
print(fire)

#get the water clusters and print
water = get_clusters('poke/water/vaporeon.png')
print("water:")
print(water)

#create the color map

#change_colors(fire, water, 'poke/water/vaporeon.png')



