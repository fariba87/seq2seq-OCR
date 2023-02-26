import tensorflow as tf
import numpy as np
import cv2
from ConFig.Config import ConfigReader
cfg = ConfigReader()


#from DataLoad.TrainDataLoading import  getDataByGenerator #GetData, data_genera2.0 *tor,
from Models.Transformer_model import OverallModel_Transformer , FULL_FE_TRANS
from Models.modules.Losses import loss_function_transformer, accuracy_function_transformer
from Models.modules.callbacks import earlystopping, lr_scheduler_tr, checkpoint_tr, backup_ckpt_tr,tensorboard_cb_tr# #tensorboard_cb , CustomSchedule
from utils import  create_in_out_decoder
from lr_schedule import cb_warmup,get_optimizer
import os
from data_generator import getDataByGenerator ,data_generator
from Models.modules.callbacks import filepath_tr, CHECKPOINT_DIR_tr
def get_data_and_model(dataset = 'MJsyn', mode ="Transformer"):

    data_gen, Maxlen, lenvoc ,vocab = getDataByGenerator(dataset= dataset ,mode =mode )#'MJsyn')
    w_max=2000
    #fullmodel = FULL_FE_TRANS(w_max, lenvocab=lenvoc + 2, maxlen=Maxlen + 2, warmup=True)#True)
    #model_fe_tr = fullmodel().model
    model = OverallModel_Transformer(w_max, lenvocab=lenvoc + 2, maxlen=Maxlen + 2,
                                     warmup=False)  # True)  #use this one for model
    #return data_gen , model_fe_tr ,fullmodel , Maxlen,lenvoc
    return data_gen , model , Maxlen,lenvoc

#data_gen , model ,fullmodel, Maxlen , lenvoc = get_data_and_model()
data_gen , model , Maxlen , lenvoc = get_data_and_model()

w_max = 2000
#
#optimizer = get_optimizer(891000)
##model  = model1().model # , cfg  = model_and_data_and_config()

##########################################################################################################
def apply_gradient(model , x, y):
    with tf.GradientTape() as tape:
        logits = model(x)#,yin)
        loss_val = loss_function_transformer(real =y, pred =logits)# tar_real
        accuracy = accuracy_function_transformer(real=y , pred=logits)#tar_real
        # print('accuracy', accuracy)
        variables = model.trans.trainable_variables + model.FEnew.trainable_variables  #model.trans :transformer - model1: embedding
        #variables = model.trainable_variables  #and change it to above
        gradients = tape.gradient(loss_val ,variables)
        #gradients = tf.clip_by_value(gradients, -1., 1.)
        gradients, _ = tf.clip_by_global_norm(gradients, 1)#0.5)
        #fullmodel.optimizer.apply_gradients(zip(gradients, variables))
        optimizer.apply_gradients(zip(gradients, variables))
    return logits, loss_val, accuracy
##########################################################################################################
optimizer = tf.keras.optimizers.Adam(0.00001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#optimizer = get_optimizer(1)#891000)
numstep = np.int32(np.divide(data_gen.__len__(), cfg.batchSize))
def train_data_for_one_epoch():
    losses =[]
    accBatch =[]
    for step in range(1):#numstep):#56):#, (x_batch_train , y_batch_train) in enumerate(trainDataset):
        #print('step:{}'.format(step))
        Xin, yAtt, yCTC, encoder_mask, times, w_max, target_weight = data_gen.__getitem__(step)
        if Xin.shape[2]>2000:
            continue
        yin, tar_real = create_in_out_decoder(yAtt, lenvoc)
        x_batch_train = (Xin, yin, encoder_mask)#X_batch_resized , y_batch_resized_Attn, encoder_mask)
        y_batch_train = tar_real#y_batch_resized_Attn
        logits, loss_val, acc = apply_gradient(model, x_batch_train, y_batch_train)
        losses.append(loss_val)
        accBatch.append(acc)
    return losses, accBatch
##########################################################################################################
_callbacks = [earlystopping, lr_scheduler_tr ,checkpoint_tr,backup_ckpt_tr ,cb_warmup]#, tensorboard_cb_tr ]#, tensorboard_cb]
print(checkpoint_tr )
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
logs = {}
train_loss_history=[]
train_acc_history =[]
val_acc_history =[]
val_loss_history =[]

def valid_data_for_one_epoch():
    losses = []
    accBatch = []
    for step in range(1):  # numstep):#56):#, (x_batch_train , y_batch_train) in enumerate(trainDataset):
        # print('step:{}'.format(step))
        Xin, yAtt, yCTC, encoder_mask, times, w_max, target_weight = data_gen.__getitem__(step)
        if Xin.shape[2] > 2000:
            continue
        yin, tar_real = create_in_out_decoder(yAtt, lenvoc)
        x_batch_train = (Xin, yin, encoder_mask)  # X_batch_resized , y_batch_resized_Attn, encoder_mask)
        y_batch_train = tar_real
        x = x_batch_train
        y = y_batch_train
        y_batch_train = tar_real  # y_batch_resized_Attn
        logits = model(x)  # ,yin)
        loss_val = loss_function_transformer(real=y, pred=logits)  # tar_real
        accuracy = accuracy_function_transformer(real=y, pred=logits)  # tar_real
        return loss_val , accuracy


def train_transformer(epochs):
    for epoch in range(epochs):#cfg.TotalEpoch):
        #callback
        callbacks.on_epoch_begin(epoch, logs=logs)


        losses_train, acc_train = train_data_for_one_epoch()
        acc_train_mean = np.mean(acc_train)
        loss_train_mean= np.mean(losses_train)
        train_loss_history.append(loss_train_mean)
        train_acc_history.append(acc_train_mean)

        losses_val, acc_val = valid_data_for_one_epoch()
        acc_val_mean = np.mean(acc_val)
        loss_val_mean = np.mean(losses_val)
        val_loss_history.append(loss_val_mean)
        val_acc_history.append(acc_val_mean)

        #losses_train_mean = np.mean(losses_train)
        #train_loss_history.append(losses_train_mean)

        print('epoch {} :loss {} and accuracy {}'.format(epoch + 1, loss_train_mean, acc_train_mean))  # , acc_train_mean)
        print('epoch {} :val-loss {} and val-accuracy {}'.format(epoch + 1, loss_val_mean, acc_val_mean))  # , acc_train_mean)
        #print('accuracy in epoch {} is {}'.format(epoch + 1, acc_train_mean))
    return train_loss_history, train_acc_history# losses_train_mean
train_loss_history, train_acc_history = train_transformer(cfg.TotalEpoch)