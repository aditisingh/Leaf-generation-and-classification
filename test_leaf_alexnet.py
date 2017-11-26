from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from scipy.cluster.hierarchy import dendrogram, linkage
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import sys, pickle
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
import os, glob
from shutil import copyfile
import scipy.io as sio
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pylab
import itertools
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import random
import math
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.misc import imresize

# Building 'AlexNet'
#Image Augmentation
img_aug=ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_blur(sigma_max=3.)
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_flip_updown()

network = input_data(shape=[None, 227, 227, 3], data_augmentation=img_aug)
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)

model = tflearn.DNN(network)
model.load('model_alex_full.tflearn')

path_vein='/uhpc/roysam/aditi/GAN/new_data/Image_prepared/GANinput_vein/'
path_border='/uhpc/roysam/aditi/GAN/new_data/Image_prepared/GANinput_border/'

vein_list=os.listdir(path_vein)
border_list=os.listdir(path_border)

vein_pred=[]
vein_label=[]

#testing on vein images 
for img in vein_list:
    infile=path_vein+img
    im=Image.open(infile)
    im1=im.crop((256,0,512,256))
    im=imresize(im1,(227,227,3))
    vein_det=model.predict(im.reshape(1,227,227,3))
    vein_pred.append(str(vein_det[0].index(max(vein_det[0]))))
    if img[:3]=='Elm':
        vein_label.append('0')
    if img[:3]=='Mag':
        vein_label.append('1')
    if img[:3]=='Map':
        vein_label.append('2')
    if img[:3]=='Oak':
        vein_label.append('3')
    if img[:3]=='Pin':
        vein_label.append('4')

print('Vein accuracy score:')
print(accuracy_score(vein_label,vein_pred)) #17.61%

border_pred=[]
border_label=[]


#testing on border images 
for img in border_list:
    infile=path_border+img
    im=Image.open(infile)
    im1=im.crop((256,0,512,256))
    im=imresize(im1,(227,227,3))
    border_det=model.predict(im.reshape(1,227,227,3))
    border_pred.append(str(border_det[0].index(max(border_det[0]))))
    if img[:3]=='Elm':
        border_label.append('0')
    if img[:3]=='Mag':
        border_label.append('1')
    if img[:3]=='Map':
        border_label.append('2')
    if img[:3]=='Oak':
        border_label.append('3')
    if img[:3]=='Pin':
        border_label.append('4')

print('Border accuracy score:')
print(accuracy_score(border_label,border_pred)) #21.37%
