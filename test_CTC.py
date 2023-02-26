#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
#export CUDA_VISIBLE_DEVICES=1
import tensorflow as tf
import numpy as np
from Models.CRNN_GCN_CTC import CTC_model
from Models.modules.Losses import loss_and_acc_function, CTCLoss
#############################################################################################################

# np.save('Adj_mat_list.npy', Adj_mat_list)
####################################################################################################################################
#for sanity check
ctc_label = np.array([ 0,  2, 42, 59, 68, 55, 54, 64, 55, 69, 69,  2], dtype=np.int32)
ctc_label_extend = np.array([[ 0,  2, 42, 59, 68, 55, 54, 64, 55, 69, 69,  2]+[77]*(25-12)], dtype=np.int32)
vocab = {' ', '!', '"', '&', "'", '(', ')', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', '@', 'A', 'B', 'C', 'D', 'E',
         'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
char2int_CTC = {' ': 0, '!': 1, '"': 2, '&': 3, "'": 4, '(': 5, ')': 6, '-': 7, '.': 8, '/': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14,
                '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, '?': 21, '@': 22, 'A': 23, 'B': 24, 'C': 25, 'D': 26, 'E': 27, 'F': 28, 'G': 29, 'H': 30, 'I': 31,
                'J': 32, 'K': 33, 'L': 34, 'M': 35, 'N': 36, 'O': 37, 'P': 38, 'Q': 39, 'R': 40, 'S': 41, 'T': 42, 'U': 43, 'V': 44, 'W': 45, 'X': 46, 'Y': 47, 'Z': 48, '[': 49,
                ']': 50, 'a': 51, 'b': 52, 'c': 53, 'd': 54, 'e': 55, 'f': 56, 'g': 57, 'h': 58, 'i': 59, 'j': 60, 'k': 61, 'l': 62, 'm': 63, 'n': 64, 'o': 65, 'p': 66,
                'q': 67, 'r': 68, 's': 69, 't': 70, 'u': 71, 'v': 72, 'w': 73, 'x': 74, 'y': 75, 'z': 76}
impath = '/media/Archive4TB3/Data/textImages/EN_Benchmarks/IC13/Word Recognition/Challenge2_Test_Task3_Images/word_1.png'
Hmax =1410
Wmax=2155
mxlen=25
len_vocab = 77
# (181, 891, 3)
wres = (2155/1410)*64  #97.815
wt= int(np.ceil(wres)) #98
SeqDivider=4

import cv2
img = cv2.imread(impath)

h, w ,_= img.shape  #181, 891
ht =64
wnew = np.int32(np.ceil((ht/h)*w))  #316
img = (cv2.resize(img, (wnew, ht))/255.)-0.5

#img= np.expand_dims(img, axis=0)
maxW=600
X = np.zeros((1, ht, maxW, 3))
X[0][:,:img.shape[1] , :]= img
t = np.ceil(img.shape[1] / SeqDivider)  #79
t=70
####################################################################################################################################
def apply_gradient(optimizer, model, x, y, times_in):
    with tf.GradientTape() as tape:
        logits = model([x, y, times_in])
        loss_val, acc = loss_and_acc_function(real =y, pred =logits, times_in = times_in)# times_in)
        #print(acc)
       # loss_val = CTCLoss(y_true=y, y_pred=logits, times=times_in)
        variables = model.trainable_variables
    gradients = tape.gradient(loss_val, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return logits, loss_val

def train_data_for_one_epoch():
    losses =[]
    for step , (x_batch_train , y_batch_train, times) in enumerate(all_ds_in_out):
        logits , loss_val = apply_gradient(optimizer , model_gctc , x_batch_train, y_batch_train, times)
        losses.append(loss_val)
    return losses


#all_ds_in_out = [(x_sanity, y_sanity , times_in)]
all_ds_in_out = [(X, ctc_label_extend, t)]
#GCTCInstance.forward_CTC()
model_gctc = CTC_model()#GCTCInstance.model_CTC_ok


optimizer = tf.keras.optimizers.Adam()
#training loop
epochs=200
for epoch in range(epochs):
    losses_train= train_data_for_one_epoch()

    #loss_val = validation()
    losses_train_mean = np.mean(losses_train)
    print(losses_train_mean)
    #loss_val_mean = np.mean(loss_val)
#tf.keras.utils.plot_model()