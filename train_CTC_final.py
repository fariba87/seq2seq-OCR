import cv2
import numpy as np
import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
################################################################################################################
################################################################################################################
from ConFig.Config import ConfigReader
cfg = ConfigReader()

from Models.modules.Losses import lossCTC , acc_function_CTC
from Models.modules.Losses import AccuracyCTC, CTCMetric, create_acc_fn
#ctcmetric = CTCMetric()

from Models.modules.Layers import  ADJlayer

from data_generator import getDataByGenerator


from Models.CRNN_GCN_CTC import modelCTC
####################################################################################
def get_data_and_model(dataset = 'MJsyn', mode ="ctc2"):

    data_gen, Maxlen, lenvoc, vocab = getDataByGenerator(dataset= dataset ,mode =mode )#'MJsyn')
    model_CTC = modelCTC(lenvoc=lenvoc)
    # model1.model
    model = model_CTC().model
    return data_gen , model_CTC ,model , Maxlen,lenvoc

data_gen , model_CTC ,model, Maxlen , lenvoc = get_data_and_model()
accCTC = AccuracyCTC(lenvocab=lenvoc)#len(vocab))  # input_length=inputlength, lenvocab)
#######################################################################################################
def apply_gradient( model, x, y):
    with tf.GradientTape() as tape:
        x1,y1, i1, l1 =x

        logits = model([x1,y1,i1,l1])#x1)#,y1)
        #print((logits.dtype))
        loss_val = lossCTC(y_true =y, y_pred =logits, i1=i1, l1=l1, lenvoc=lenvoc)# times_in)
        acc_val = accCTC(y_true =y, y_pred =logits)
        #acc_val = acc_function_CTC(y_true =y, y_pred =logits)
        variables = model.trainable_variables
    gradients = tape.gradient(loss_val, variables)
    model_CTC.optimizer.apply_gradients(zip(gradients, variables))
    # print(getMaxListEagerTensores(gradients))
    return logits, loss_val  , acc_val#, acc_val


#import logging
import os
#logging.basicConfig(level=logging.DEBUG)
# def load_model(self, model_path=None):
#     if model_path == None:
#         latest = tf.train.latest_checkpoint(checkpoint_dir)
#         if latest != None:
#             logging.info('load model from {}'.format(latest))
#             last_epoch = int(latest.split('-')[-1])
#             checkpoint.restore(latest)
#     else:
#         logging.info('load model from {}'.format(model_path))
#         checkpoint.restore(model_path)


#@tf.function
numstep =np.int0( 891927/8)#np.int32(np.divide(data_gen.__len__(), cfg.batchSize))
def train_data_for_one_epoch():
    #load_model()

    lossBatch =[]
    accBatch =[]
    for step in range(1):#numstep):#00):#numstep): #numstep # , (x_batch_train , y_batch_train) in enumerate(all_ds_in_out):
        if step % 10000 == 0:
            print(step)
        [X, yCTC, inputlength, labellenght], yCTC = data_gen.__getitem__(step)
        if X.shape[2]>2000:
            continue
        x_batch_train = X, yCTC, inputlength, labellenght
        y_batch_train = yCTC #tf.ragged.constant(yCTC) #yCTC
        #model = build_model(lenvoc=lenvoc, input_shape=X.shape[1:], label_shape=yCTC.shape[1])
        logits , loss_val , acc_val = apply_gradient(model , x_batch_train, y_batch_train)
        #print(loss_val)
        #print(acc_val)
        lossBatch.append(loss_val)
        accBatch.append(acc_val)
    return lossBatch , accBatch


epochs=cfg.TotalEpoch
loss =[]
acc =[]
from Models.modules.callbacks import earlystopping, lr_scheduler, tensorboard_cb_ctc , checkpoint_ctc, backup_ckpt_ctc
_callbacks = [earlystopping, lr_scheduler , tensorboard_cb_ctc, checkpoint_ctc,backup_ckpt_ctc ]
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
logs = {}
def trainCTC():
    for epoch in range(epochs):
        #print(epoch)
        callbacks.on_epoch_begin(epoch, logs=logs)
        losses_train , acc_train = train_data_for_one_epoch()
        losses_train_mean = np.mean(losses_train)
        acc_train_mean = np.mean(acc_train)
        loss.append(losses_train_mean )
        acc.append(acc_train_mean)
        print('loss and accuracy in epoch {} is {} and {}'.format(epoch+1 , losses_train_mean, acc_train_mean))# , acc_train_mean)
    return losses_train_mean, acc_train_mean
losses_train_mean, acc_train_mean = trainCTC()
#######################################################################################################
# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(loss)
# plt.title('loss')
# plt.subplot(1,2,2)
# plt.plot(acc)
# plt.title('accuracy')
# plt.show()
