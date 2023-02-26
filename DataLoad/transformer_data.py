#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
#export CUDA_VISIBLE_DEVICES=1

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ConFig.Config import ConfigReader
cfg = ConfigReader()
from transformer import Transformer
dftest = pd.read_csv('/media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/sample_df.csv')
#Index(['Unnamed: 0', 'impath', 'label', 'Hmax', 'Wmax', 'Attnlabel'], dtype='object')
paths= tf.convert_to_tensor(dftest['impath'])
Wmax = max(pd.unique(dftest['Wmax']).tolist())  #2155
Hmax = max(pd.unique(dftest['Hmax']).tolist())  #1410
Htarget = cfg.targetHeight
Wtarget = np.ceil((Wmax/Hmax)*Htarget)

seq_len = int(np.ceil(Wtarget  / cfg.SeqDivider))
times = tf.cast(tf.tile(tf.expand_dims(seq_len, 0), [32]), dtype=tf.int32)  #32

Lenmax = maxlen =27 # for this dataset

#########################################################################

import tensorflow as tf
from keras import layers
from keras.initializers.initializers_v2 import GlorotNormal, RandomUniform

initializer = RandomUniform(minval=-0.50, maxval=0.50)
# initializer = GlorotNormal()

#model1 =mobilenet()
#############################################################################
#path1 = '/media/Archive4TB3/Data/textImages/EN_Benchmarks/IC13/Word Recognition/Challenge2_Test_Task3_Images/word_1.png'
#label1 = "Tiredness"
#tf.py_function(func=load_image, inp=[image_path,imageormask ], Tout=tf.float32, name=None)
def load_image(image_path):
    image_path = image_path.numpy()
    image = cv2.imread(image_path)
    [h,w,c] = image.shape
    image = cv2.resize(image, (int((w*64)/h), 64))
    im =np.concatenate([image , np.zeros((64,85,1 ))], axis =1)
    return image

  #target_size = [64, maxW] target_size=(64,160),target_size=(64,400)
  # image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')#, target_size=(32, 32),
  #                                              color_mode='grayscale')
  #   image = tf.keras.preprocessing.image.img_to_array(image)
  #tf.enable_eager_execution()
  # image_path = image_path.numpy()
  #  image_path = image_path.decode('utf-8')
########################################################################################
#################### create dataset for images  ####################
def load_image_mask(image_path, maxW=411, imageormask=True):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    #image = tf.image.resize(image, (target_size))
    w =tf.shape(image)[1]
    h= tf.shape(image)[0]
   # scale = 64/h
   # nw = scale*w
 #image.get_shape()#shape#tf.shape(image)#.get_shape()#
    #tf.print('')
    #img = cv2.resize(img, (int(w * 64 / h), 64))

    image = tf.image.resize(image, (cfg.targetHeight, tf.math.ceil(tf.divide(tf.multiply(w , cfg.targetHeight) ,  h)))) # with respect to hmax, wmax in that dataset
    [h, w, c] = image.shape# get_shape()#shape#tf.shape(image)  #image.get_shape()#tf.shape(image)
    # tf.shape(image) [w, h, c]
   # image = tf.image.resize(image,(htarget, w) )
    w = tf.shape(image)[1]
    padsize = maxW- w+1
    #tf.pad()
    #image = tf.expand_dims(image, axis = 0)
    paddings = [[0, 0], [0, padsize],[0,0]]
    image = tf.pad(image, paddings, "CONSTANT")
    image = tf.cast(image, tf.float32) / 255.
    image = tf.expand_dims(image, axis=0)  #maybe it is not needed
    #image = tf.keras.layers.ZeroPadding2D(padding=((0,0   ), (0,padsize )), data_format=None)(image)
    #image = tf.squeeze(image, axis = 0)
    #encodermask = tf.concat([tf.zeros((1, maxW-w)), tf.ones((1, w))], axis = 1) # 1,400
    #
    #image = tf.image.resize(image, target_size)
    #if imageormask :
    return image
    #else :
        #sample_image = sample_image.numpy()
    #    return encodermask#, label , image

image_ds = tf.data.Dataset.from_tensor_slices(dftest['impath'].values.tolist()[:1]).map(load_image_mask)#, num_parallel_calls=AUTOTUNE)
########################################################################################
#################### create dataset for masks  ####################
def load_mask(image_path, maxW=400):
    dev = cfg.SeqDivider
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    w = tf.shape(image)[1]
    h = tf.shape(image)[0]
    #htarget = 64
    image = tf.image.resize(image, (cfg.targetHeight, tf.math.ceil(tf.divide(tf.multiply(w, cfg.targetHeight), h))))  # with respect to hmax, wmax in that dataset
    [h, w, c] = image.shape  # get_shape()#shape#tf.shape(image)  #image.get_shape()#tf.shape(image)
    w = tf.shape(image)[1]
    padsize = tf.cast(tf.divide(maxW - w , dev), tf.int32)
    wnew = tf.cast(tf.divide(w , dev), tf.int32)
    encodermask = tf.concat([tf.zeros((1,padsize)), tf.ones((1, wnew+2))], axis=1)
    #encodermask = tf.concat([tf.zeros((1, maxW-w)), tf.ones((1, w))], axis=1)  # 1,400
    return encodermask

#dftest = dftest.iloc[0:2]
encmask_ds = tf.data.Dataset.from_tensor_slices(dftest['impath'].values.tolist()[:1]).map(load_mask) # argument = False
########################################################################################
########################### create dataset for labels  #################################
def convertbacktonumpy(A):  #AttnLabel column of DF was save as string
    AA = A[1:-1]
    return np.fromstring(AA, dtype=int, sep=' ')
def paddedseq(label, maxlen=27):
     len_label= len(label)
     padsize = maxlen- len_label
#     paddings = [[0, padsize]]# , [0, padsize], [0, 0]]
#     paddedseq= tf.pad(label, paddings, "CONSTANT", constant_values=-1)
#     paddedseq = np.concatenate([label , np.zeros(padsize)-1])
     paddedseq = label.tolist()+[0]*padsize  # 0 or -1 : here we use 0 padding
     return np.array(paddedseq)
# one mistake that was happened is that during load of df this column has cconverted to str intead on numpy
numpylabels = list(map(convertbacktonumpy, dftest['Attnlabel'].values.tolist()))
listpadded = list(map(lambda p : paddedseq(p, maxlen =19), numpylabels))

label_ds = tf.data.Dataset.from_tensor_slices(listpadded[:1] )
for i in label_ds.take(1):
    print(i)
########################################################################################
#################### create dataset for (x,y) for model fitting  ########################
all_ds_in = tf.data.Dataset.zip((image_ds, label_ds, encmask_ds))
all_ds_in_out = tf.data.Dataset.zip((all_ds_in , label_ds))
batchsize = 1
all_ds_in_out =all_ds_in_out.batch(batchsize)