import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from utils import convertbacktonumpy ,paddedseq, encode_txt2ind
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
################################################################################################################
from ConFig.Config import ConfigReader
cfg = ConfigReader()
dataloaderDS =['transformer' , 'CTC']
vocab_CTC = {'0', '1', '5', '7', '8', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
Maxlen_MJ= 19
lenvocab=57

################################################################################################################
class data_generator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = cfg.batchSize,
                 shuffle=False,
                 H=cfg.targetHeight,
                 CTC=False ,
                 Attn=False,
                 char2int_Attn=1 ,
                 char2int_CTC=1,
                 seqDivider =cfg.SeqDivider,
                 num_channel =1,
                 padCTC = lenvocab,
                 padTran =0):
        self.df = df
        self.batch_size = batch_size
        self.HTraget = H
        self.number_of_samples = len(self.df)
        self.CTC = CTC
        self.Attn = Attn
        self.char2int_Attn= char2int_Attn
        self.seqDivider = seqDivider
        self.numchannel =num_channel
        self.shuffle =shuffle
        self.char2int_CTC =char2int_CTC
        self.padCTC = padCTC
        self.padTran = padTran

    def __getitem__(self, index):
        batches = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]  # .iloc[]
        X, yAtt,yCTC, mask , t ,w_max= self.__get_data(batches)  # or y1 and y2
        # if self.CTC:
        #     return (X, yCTC), yCTC
        # if self.Transformer:
        #     return (X, mask,yAtt ) , yAtt
        return X, yAtt,yCTC , mask , t , w_max  # return a complete batch

    def process_image(self, Hmax, Wmax):  # , H, W):  # there is a problem here! since for scale i got 0
        img = cv2.imread(self.impath)# scale w based on max W in that ds or alllllll?!
        h, w = img.shape[:2]
        #img = cv2.resize(img, ())
       # B = np.int0((w/h)*64)
       # A= np.int0((Wmax/Hmax)*64)
       # C= np.int0((A/B)*64)

        scale = self.HTraget / h
        nw = scale * w
        #img = cv2.resize(img, (nw, H)) / 255.
        #img = cv2.resize(img, (C, A)) #/ 255.
        #img = cv2.resize(img, (64, A)) / 255.
        img = np.array(cv2.resize(img, (np.int0(nw),self.HTraget))) / 255.
        h1, w1, _ = img.shape
        t = np.ceil(w1/ self.seqDivider)
        if np.ndim(img) != 3:
            img = np.expand_dims(img, axis=-1)
        return img ,w1,t #nw

    def __get_data(self, batches):
        colnames = ['impath', 'label' , 'Hmax' , 'Wmax' , 'CTClabel' , 'Attnlabel'] # two other column (for ctc and attn label) two other column hmax, wmax
        img_path_list = batches[colnames[0]].to_numpy().tolist()
        text_list = batches[colnames[1]].to_numpy().tolist()
        Hmax_ds = batches[colnames[2]].to_numpy().tolist()
        Wmax_ds = batches[colnames[3]].to_numpy().tolist()
        text_list_Encoded_CTC   = batches [colnames[4]]
        text_list_Encoded_Attn  = batches [colnames[5]]
        w_max = 1
        max_seq_len = 1
        x_batch = []
        y_batch = []
        times =[]
        for i in range(self.batch_size):
            #   im_lab=image_and_label_numpy[i]
            #  im_path, label=im_lab.split('\t')[:2]
            self.impath = img_path_list[i]
            img, w , t = self.process_image( Hmax_ds[i] , Wmax_ds[i] )
            times.append(t)
          #  text_list[i]
          #  label = 'sos' + label + 'eos' #!
            # need label encoding

            if w > w_max:
                w_max = np.int0(w)
            if len(text_list[i]) > max_seq_len:
                max_seq_len = len(text_list[i])  # max_len_per_bacth #text_list_Encoded_CTC

            x_batch.append(img)
            y_batch.append(np.array([self.char2int_Attn_syn.tolist()[j] for j in text_list[i]]))


        X_batch_resized = np.zeros((self.batch_size, self.H, w_max, self.numchannel), dtype=np.float32)

        y_batch_resized_CTC  = np.zeros((self.batch_size, max_seq_len), dtype=np.float32)+self.padCTC  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        y_batch_resized_Attn = np.zeros((self.batch_size, max_seq_len+2), dtype=np.float32)+self.padTran  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        encoder_mask = np.zeros((self.batch_size, np.int32(np.divide(w_max,self.seqDivider))))

        for i in range(self.batch_size):
            X_batch_resized[i, :, :x_batch[i].shape[1], :] = x_batch[i]
            y_batch_resized_CTC[i,  :len(convertbacktonumpy(text_list_Encoded_CTC[i]))] = convertbacktonumpy(text_list_Encoded_CTC.iloc[i])  # y_batch[i]
            y_batch_resized_Attn[i, :len(convertbacktonumpy(text_list_Encoded_Attn.iloc[i]))] = convertbacktonumpy(text_list_Encoded_Attn.iloc[i])
            #encoderInputMask[i][int(sourceImageWidth / self.hparams.sourceEmbeddingDiv):] = 1
            encoder_mask[i][np.int32(np.divide(x_batch[i].shape[1],self.seqDivider)):] = 1# np.ones_like((1,x_batch[i].shape[1])

        #if self.CTC :
         #   return X_batch_resized, y_batch_resized_CTC
        #else :
         #   return X_batch_resized, y_batch_resized_Attn  # y_batch_resized  # y_batch_ctc , y_batch_attn
    #    y_batch_resized_CTC[i, :len(y_batch[i])] = y_batch[i]
        return X_batch_resized, y_batch_resized_Attn, y_batch_resized_CTC  ,encoder_mask , np.array(times), w_max
###############################################################################################################

# one mistake that was happened is that during load of df this column has cconverted to str intead on numpy

maxW =399#600 #127
###############################################################################################################
def load_image_mask(image_path, maxW=maxW, imageormask=True): #411 , 2619
    dev = cfg.SeqDivider
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    # image = tf.image.resize(image, (target_size))
    w = tf.shape(image)[1]
    h = tf.shape(image)[0]
    image = tf.image.resize(image, (cfg.targetHeight, tf.math.ceil(
        tf.divide(tf.multiply(w, cfg.targetHeight), h))))  # with respect to hmax, wmax in that dataset
    [h, w, c] = image.shape

    image = tf.image.resize(image, (cfg.targetHeight,400))#cfg.targetHeight,maxW)) # i added recently
    [h, w, c] = image.shape  # get_shape()#shape#tf.shape(image)  #image.get_shape()#tf.shape(image)
    # tf.shape(image) [w, h, c]
    # image = tf.image.resize(image,(htarget, w) )
    w = tf.shape(image)[1]

    padsize = maxW - w + 1
    # tf.pad()
    # image = tf.expand_dims(image, axis = 0)
    paddings = [[0, 0], [0, padsize], [0, 0]]
    image = tf.pad(image, paddings, "CONSTANT")
    image = tf.cast(image, tf.float32) / 255.
    #image = tf.expand_dims(image, axis=0)  # maybe it is not needed
    #added
    padsize = tf.cast(tf.divide(maxW - w , dev), tf.int32)
    wnew = tf.cast(tf.divide(w , dev), tf.int32)
    encodermask = tf.concat([tf.zeros((1,padsize)), tf.ones((1, wnew))], axis=1)

    return image , encodermask


def get_syn_DS(DF , data, N_sample, batchsize , chars):#cfg.batchSize): #-1, #number of samples to use in model
    DFSyn = pd.read_csv('df_train_syn.csv') #6965052
    # DF.keys() #['Unnamed: 0', 'impath', 'label', 'Hmax', 'Wmax', 'CTClabel','Attnlabel']
    Wmax = max(pd.unique(DFSyn['Wmax']).tolist())  # 2619
    Hmax = max(pd.unique(DFSyn['Hmax']).tolist())  # 600
    char2int_Attn_syn = np.load('char2int_Attn_syn.npy', allow_pickle=True)
    char2int_CTC_syn = np.load('char2int_CTC_syn.npy', allow_pickle=True)
    Maxlen_Syn = np.load('mx1_syn.npy')  # 61
    vocab_Attn = list(char2int_Attn_syn.tolist().keys())
    vocab_CTC = list(char2int_CTC_syn.tolist().keys())
    image_encodermask_ds = tf.data.Dataset.from_tensor_slices(DFSyn['impath'].values.tolist()[:N_sample]).map(load_image_mask)
    if data == dataloaderDS[0]:
        image_ds = image_encodermask_ds.map(lambda a, b: a)
        encmask_ds = image_encodermask_ds.map(lambda a, b: b)

#        numpylabels = list(map(convertbacktonumpy, DFSyn['Attnlabel'].values.tolist()))
        numpylabels = list(DFSyn['label'].values.tolist())
        V = list(map(lambda p: encode_txt2ind(p, chars), numpylabels))

        listpadded = list(
            map(lambda p: paddedseq(p, maxlen=Maxlen_Syn, defal=0), numpylabels))  # paddedseq has an argument that should be set
        label_ds = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])

        all_ds_in = tf.data.Dataset.zip((image_ds, label_ds, encmask_ds))
        all_ds_in_out = tf.data.Dataset.zip((all_ds_in, label_ds))
        #batchsize = 8#cfg.batchSize
        all_ds_in_out_syn =all_ds_in_out.batch(batchsize,  drop_remainder=True) # shuffle()
    else :
        #image_encodermask_ds = tf.data.Dataset.from_tensor_slices(DFSyn['impath'].values.tolist()[:100]).map(load_image_mask())
        image_ds = image_encodermask_ds.map(lambda a, b: a)
        numpylabels = list(map(convertbacktonumpy, DFSyn['CTClabel'].values.tolist()))
        listpadded = list(
            map(lambda p: paddedseq(p, maxlen=Maxlen_Syn, defal = len(vocab_Attn)), numpylabels))  # paddedseq has an argument that should be set
        label_ds = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in = tf.data.Dataset.zip((image_ds, label_ds))#, encmask_ds))
        all_ds_in_out = tf.data.Dataset.zip((all_ds_in, label_ds))
        #batchsize = 256  # cfg.batchSize
        all_ds_in_out_syn = all_ds_in_out.batch(batchsize, drop_remainder=True)  # shuffle()
    return all_ds_in_out_syn #.cache().prefetch(tf.data.experimental.AUTOTUNE)


###############################################################################################################
def get_MJ_DS(DF , data,  N_sample,  batchsize , chars): #cfg.batchSize): #-1. 100
   # DFMJ = pd.read_csv('df_train_MJ1000.csv')
    DFMJ = DF
    image_encodermask_ds_MJ = tf.data.Dataset.from_tensor_slices(DFMJ['impath'].values.tolist()[:N_sample]).map(
        load_image_mask)
    image_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: a)
    # numpylabels = list(map(convertbacktonumpy, DFMJ['Attnlabel'].values.tolist()))
    numpylabels = list(DFMJ['label'].values.tolist())
    V = list(map(lambda p: encode_txt2ind(p, chars), numpylabels))
    #data = 'CTC'
    if data==dataloaderDS[0]:


        encmask_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: b)

        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen_MJ, defal=0), numpylabels))
        label_ds_MJ = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in_MJ = tf.data.Dataset.zip((image_ds_MJ, label_ds_MJ, encmask_ds_MJ))
        all_ds_in_out_MJ = tf.data.Dataset.zip((all_ds_in_MJ, label_ds_MJ))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out_MJ = all_ds_in_out_MJ.batch(batchsize, drop_remainder=True)# shuffle()
    else :
        #image_encodermask_ds_MJ = tf.data.Dataset.from_tensor_slices(DFMJ['impath'].values.tolist()[:N_sample]).map(load_image_mask)
        #image_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: a)
        #numpylabels= list(map(convertbacktonumpy, DFMJ['CTClabel'].values.tolist()))
        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen_MJ, defal = len(vocab_CTC)), numpylabels))  #19. 57
        label_ds_MJ = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in_MJ = tf.data.Dataset.zip((image_ds_MJ, label_ds_MJ))#, encmask_ds_MJ))
        all_ds_in_out_MJ = tf.data.Dataset.zip((all_ds_in_MJ, label_ds_MJ))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out_MJ = all_ds_in_out_MJ.batch(batchsize, drop_remainder=True)# shuffle()
    return all_ds_in_out_MJ
  #paddedseq has an argument that should be set
def GetData(dataset, ctc=True):
    if dataset == 'Syntext':
        DFSyn = pd.read_csv('df_train_syn.csv')
        char2int = np.load('char2int_CTC_syn.npy', allow_pickle=True)
        Maxlen = np.load('mx1_syn.npy')
        if ctc :
            vocab = list(char2int.tolist().keys())
            #char_to_idx = {}
            #idx_to_char = {}
            chars_Attn = vocab

        else:
            vocab = ['SOS']+list(char2int.tolist().keys())+['EOS']
            #char_to_idx = {"PAD": 0}
            #idx_to_char = {0: "PAD"}
            chars_Attn = ["PAD"] + vocab  # ,"SOS"]+vocab+["EOS"]
            Maxlen = Maxlen + 2
        chars = {ch: i for i, ch in enumerate(chars_Attn)}
        char_array_Attn = np.array(chars_Attn)
        get_syn_DS(DF=DFSyn, data = 'CTC', N_sample = 1, batchsize = 1, chars=chars)
        DF = DFSyn
        #get_syn_DS(DFSyn , data = 'CTC', N_sample =1000, batchsize =8, chars=chars)
    else:
        DFMJ = pd.read_csv('df_MJsyn')
        Maxlen = np.load('mx1_MJsyn.npy')
        Maxlen = Maxlen + 2
        vocab = np.load('vcMJsyn.npy', allow_pickle=True)
        vocab = ['SOS']+list(vocab.tolist())+['EOS']
        char_to_idx = {"PAD": 0}
        idx_to_char = {0: "PAD"}
        chars_Attn = ["PAD"] + vocab
        chars = {ch: i for i, ch in enumerate(chars_Attn)}
        char_array_Attn = np.array(chars_Attn)
        DF = DFMJ
        get_MJ_DS(DF= DFMJ , data ='ctc' ,  N_sample=1,  batchsize =1 , chars=chars)
    return DF , Maxlen , chars , vocab


GetData(dataset= 'Syntext')

# char2int_CTC_syn = np.load('../char2int_CTC_syn.npy', allow_pickle=True)
# vocab = ['SOS']+list(char2int_CTC_syn.tolist().keys())+['EOS']
# char_to_idx = {"PAD": 0}
# idx_to_char = {0: "PAD"}
# DFSyn = pd.read_csv('../df_train_syn.csv')
#
# chars_Attn = ["PAD" ]+vocab# ,"SOS"]+vocab+["EOS"]
# chars = {ch: i for i, ch in enumerate(chars_Attn)}
# char_array_CTC = np.array(chars_Attn)
# DFSyn['label'][i]
# text_encoded_Attn =  np.array([chars[ch] for ch in DFSyn['label'][1000]], dtype=np.int32)
# text_encoded =  np.concatenate((np.expand_dims(chars['SOS'],0),
#                 np.array([chars[ch] for ch in DFSyn['label'][1000]], dtype=np.int32),
#                 np.expand_dims(chars['EOS'], axis=0)), axis =0)
# def encode_txt2ind(x, chars):
#     text_encoded = np.concatenate((np.expand_dims(chars['SOS'], 0),
#                                    np.array([chars[ch] for ch in str(x)], dtype=np.int32),
#                                    np.expand_dims(chars['EOS'], axis=0)), axis=0)
#     return text_encoded

def get_MJ_DS(DF , data,  N_sample,  batchsize , chars): #cfg.batchSize): #-1. 100
   # DFMJ = pd.read_csv('df_train_MJ1000.csv')
    DFMJ = DF
    image_encodermask_ds_MJ = tf.data.Dataset.from_tensor_slices(DFMJ['impath'].values.tolist()[:N_sample]).map(
        load_image_mask)
    image_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: a)
    # numpylabels = list(map(convertbacktonumpy, DFMJ['Attnlabel'].values.tolist()))
    numpylabels = list(DFMJ['label'].values.tolist())
    V = list(map(lambda p: encode_txt2ind(p, chars), numpylabels))
    #data = 'CTC'
    if data == dataloaderDS[0]:
        encmask_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: b)
        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen_MJ, defal=0), numpylabels))
        label_ds_MJ = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in_MJ = tf.data.Dataset.zip((image_ds_MJ, label_ds_MJ, encmask_ds_MJ))
        all_ds_in_out_MJ = tf.data.Dataset.zip((all_ds_in_MJ, label_ds_MJ))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out_MJ = all_ds_in_out_MJ.batch(batchsize, drop_remainder=True)# shuffle()
    else :
        #image_encodermask_ds_MJ = tf.data.Dataset.from_tensor_slices(DFMJ['impath'].values.tolist()[:N_sample]).map(load_image_mask)
        #image_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: a)
        #numpylabels= list(map(convertbacktonumpy, DFMJ['CTClabel'].values.tolist()))
        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen_MJ, defal = len(vocab_CTC)), numpylabels))  #19. 57
        label_ds_MJ = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in_MJ = tf.data.Dataset.zip((image_ds_MJ, label_ds_MJ))#, encmask_ds_MJ))
        all_ds_in_out_MJ = tf.data.Dataset.zip((all_ds_in_MJ, label_ds_MJ))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out_MJ = all_ds_in_out_MJ.batch(batchsize, drop_remainder=True)# shuffle()
    return all_ds_in_out_MJ