import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Input
from keras import optimizers
from keras.models import load_model
import numpy as np
import scipy.misc
import imageio
import scipy.ndimage
import cv2
import math
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

img_shape = (32,32,1) #(None,None,1)
input_img = Input(shape=(img_shape))
#creating the architecture of srcnn model
ConvL1 = Conv2D(64,(9,9),padding='SAME',name='CONV1')(input_img)
ActivL1 = Activation('relu', name='act1')(ConvL1)
ConvL2 = Conv2D(32,(1,1),padding='SAME',name='CONV2')(ActivL1)
ActivL2 = Activation('relu', name='act2')(ConvL2)
ConvL3 = Conv2D(1,(5,5),padding='SAME',name='CONV3')(ActivL2)
ActivL3 = Activation('relu', name='act3')(ConvL3)
model = Model(input_img, ActivL3)
opt = optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=opt,loss='mean_squared_error')
model.summary()

