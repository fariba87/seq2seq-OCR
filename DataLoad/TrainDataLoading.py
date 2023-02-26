import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from utils import convertbacktonumpy ,paddedseq, encode_txt2ind
from ConFig.Config import ConfigReader
cfg = ConfigReader()
dataloaderDS =['transformer' , 'CTC']
maxW =649#600 #127
###############################################################################################################
def load_image_mask(image_path, maxW=maxW, imageormask=True): #411 , 2619
    dev = cfg.SeqDivider
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize(image, (target_size))
    [h,w,c] = image.shape.as_list()
    w = tf.shape(image)[1]
    h = tf.shape(image)[0]
    image = tf.image.resize(image, (cfg.targetHeight, tf.math.ceil(tf.divide(tf.multiply(w, cfg.targetHeight), h))))  # with respect to hmax, wmax in that dataset
    w = tf.shape(image)[1]
    padsize = tf.cast(maxW - w + 1, tf.float32)
    # tf.pad()
    # image = tf.expand_dims(image, axis = 0)
    paddings = [[0, 0], [0, padsize], [0, 0]]
   # image = tf.pad(image, paddings, "CONSTANT")
    image = tf.cast(image, tf.float32) / 255.
    #image = tf.expand_dims(image, axis=0)  # maybe it is not needed
    #added
    image = tf.concat([image, tf.zeros((cfg.targetHeight, padsize, 1))],axis=1)
    padsize = tf.cast(tf.divide(padsize , dev), tf.int32)#.numpy()
    w  = tf.cast(w, tf.float32)
    wnew = tf.cast(tf.divide(w , dev), tf.int32)#.numpy()
    encodermask = tf.concat([tf.zeros((1,padsize)), tf.ones((1, wnew))], axis=1)

    return image , encodermask
def gen_DS(DF , Maxlen , chars , vocab , Wmax, Hmax,N_sample, batchsize, ctc=True ):
    #DFMJ = DF
    image_encodermask_ds = tf.data.Dataset.from_tensor_slices(DF['impath'].values.tolist()[:N_sample]).map(
        load_image_mask)
    image_ds = image_encodermask_ds.map(lambda a, b: a)
    # numpylabels = list(map(convertbacktonumpy, DFMJ['Attnlabel'].values.tolist()))
    numpylabels = list(DF['label'].values.tolist()[:N_sample])
    V = list(map(lambda p: encode_txt2ind(p, chars), numpylabels))
    #data = 'CTC'
    if not ctc:#data== dataloaderDS[0]: #transformer


        encmask_ds = image_encodermask_ds.map(lambda a, b: b)

        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen, defal=0), V))#numpylabels))
        label_ds = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in = tf.data.Dataset.zip((image_ds, label_ds, encmask_ds))
        all_ds_in_out = tf.data.Dataset.zip((all_ds_in, label_ds))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out = all_ds_in_out.batch(batchsize, drop_remainder=True)# shuffle()
    else :
        #image_encodermask_ds_MJ = tf.data.Dataset.from_tensor_slices(DFMJ['impath'].values.tolist()[:N_sample]).map(load_image_mask)
        #image_ds_MJ = image_encodermask_ds_MJ.map(lambda a, b: a)
        #numpylabels= list(map(convertbacktonumpy, DFMJ['CTClabel'].values.tolist()))
        listpadded = list(map(lambda p: paddedseq(p, maxlen=Maxlen, defal = len(vocab)), V))#numpylabels))  #19. 57
        label_ds = tf.data.Dataset.from_tensor_slices(listpadded[:N_sample])
        all_ds_in = tf.data.Dataset.zip((image_ds, label_ds))#, encmask_ds_MJ))
        all_ds_in_out = tf.data.Dataset.zip((all_ds_in, label_ds))
        #batchsize = 8  # cfg.batchSize
        all_ds_in_out = all_ds_in_out.batch(batchsize, drop_remainder=True)# shuffle()
    return all_ds_in_out

def GetData(dataset, ctc=True, N_sample=1, batchsize=1 ):
    if dataset == 'Syntext':
        DFSyn = pd.read_csv('df_train_syn.csv')
        char2int = np.load('char2int_CTC_syn.npy', allow_pickle=True)  #93
        Maxlen = np.load('mx1_syn.npy')
        Wmax = max(pd.unique(DFSyn['Wmax']).tolist())  # 2619
        Hmax = max(pd.unique(DFSyn['Hmax']).tolist())  # 600
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
       # get_syn_DS(DF=DFSyn, data = 'CTC', N_sample = 1, batchsize = 1, chars=chars)
        DF = DFSyn
        #get_syn_DS(DFSyn , data = 'CTC', N_sample =1000, batchsize =8, chars=chars)
    elif dataset == 'MJSyn':
        DFMJ = pd.read_csv('df_MJsyn.csv')
        Maxlen = np.load('mx1_MJsyn.npy')
        Wmax = max(pd.unique(DFMJ['Wmax']).tolist())  # 2619
        Hmax = max(pd.unique(DFMJ['Hmax']).tolist())  # 600
        vocab = np.load('vcMJsyn.npy', allow_pickle=True)  #62
        if ctc:
            chars_Attn = vocab
        else :
            Maxlen = Maxlen + 2
            #vocab = np.load('vcMJsyn.npy', allow_pickle=True)
            vocab = ['SOS']+list(vocab.tolist())+['EOS']
            #char_to_idx = {"PAD": 0}
            #idx_to_char = {0: "PAD"}
            chars_Attn = ["PAD"] + vocab
        chars = {ch: i for i, ch in enumerate(chars_Attn)}
        char_array_Attn = np.array(chars_Attn)
        DF = DFMJ
        #get_MJ_DS(DF= DFMJ , data ='ctc' ,  N_sample=1,  batchsize =1 , chars=chars)
    maxW=399
    return gen_DS(DF , Maxlen , chars , vocab , Wmax, Hmax, N_sample, batchsize , ctc), vocab, Maxlen ,maxW
dataset =['Syntext', 'MJSyn']
#GetData(dataset='Syntext', ctc=True, N_sample=1, batchsize=1 )

#######################################
class data_generator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = cfg.batchSize,
                 shuffle=False,
                 H=cfg.targetHeight,
                 CTC=False ,
                 Attn=False,
                 char2int_Attn=1 ,
                 char2int_CTC=1,
                 seqDivider =cfg.SeqDivider,
                 num_channel =3,
                 vocab=['a'],
                 padCTC = 1000,
                 padTran =0):
        self.df = df
        self.batch_size = batch_size
        self.HTraget = H
        self.number_of_samples = len(self.df)
        self.seqDivider = seqDivider
        self.numchannel =num_channel
        self.shuffle =shuffle
        self.vocab=vocab
        self.padCTC = len(vocab)
        self.padTran = padTran

        self.voc_CTC= self.vocab
        self.char_CTC =self.voc_CTC
        self.voc_Att = ['SOS']+self.vocab+['EOS']
        self.chars_att=['PAD']+self.voc_Att
        self.chars_Attention = {ch: i for i, ch in enumerate(self.chars_att)}
        self.chars_CTC = {ch: i for i, ch in enumerate(self.char_CTC)}




    def __getitem__(self, index):
        batches = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]  # .iloc[]
        X, yAtt,yCTC, mask , t ,w_max= self.__get_data(batches)  # or y1 and y2
        return X, yAtt,yCTC , mask , t , w_max  # return a complete batch

    def process_image(self, Hmax, Wmax):  # , H, W):  # there is a problem here! since for scale i got 0
        img = cv2.imread(self.impath, 0)# scale w based on max W in that ds or alllllll?!
        h, w = img.shape[:2]
        scale = self.HTraget / h
        nw = scale * w
        img = np.array(cv2.resize(img, (np.int0(nw),self.HTraget))) / 255.
        h1, w1 = img.shape
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
        w_max = 1
        max_seq_len = 1
        x_batch = []
        y_batch = []
        times =[]
        for i in range(self.batch_size):
            self.impath = img_path_list[i]
            img, w , t = self.process_image( Hmax_ds[i] , Wmax_ds[i] )
            times.append(t)
            if w > w_max:
                w_max = np.int0(w)
            if len(text_list[i]) > max_seq_len:
                max_seq_len = len(text_list[i])  # max_len_per_bacth #text_list_Encoded_CTC
            x_batch.append(img)
#            y_batch.append(np.array([self.char2int_Attn_syn.tolist()[j] for j in text_list[i]]))


        X_batch_resized = np.zeros((self.batch_size, self.HTraget, w_max, 3), dtype=np.float32)

        y_batch_resized_CTC  = np.zeros((self.batch_size, max_seq_len), dtype=np.float32)+self.padCTC  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        y_batch_resized_Attn = np.zeros((self.batch_size, max_seq_len+2), dtype=np.float32)+self.padTran  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        encoder_mask = np.zeros((self.batch_size, np.int32(np.divide(w_max,self.seqDivider))))

        for i in range(self.batch_size):
            X_batch_resized[i, :, :x_batch[i].shape[1], :] = x_batch[i]
            en_Att = np.concatenate((np.expand_dims(self.chars_Attention['SOS'], 0),
                                   np.array([self.chars_Attention[ch] for ch in str(text_list[i])], dtype=np.int32),
                                   np.expand_dims(self.chars_Attention['EOS'], axis=0)), axis=0)

            en_CTC =  np.asarray([self.chars_CTC[ch] for ch in str(text_list[i])], dtype=np.int32)

            y_batch_resized_CTC[i, :len(en_CTC)] = en_CTC
            y_batch_resized_Attn[i, :len(en_Att)] =en_Att



        return X_batch_resized, y_batch_resized_Attn, y_batch_resized_CTC  ,encoder_mask , np.array(times), w_max
#X_batch_resized, y_batch_resized_Attn, y_batch_resized_CTC  ,encoder_mask , times, w_max = data_generator(df, batch_size = 16, shuffle=False, H=64,seqDivider =4,
#                 num_channel =1,  vocab=vocab, padCTC = len(vocab), padTran =0).__getitem__(0)
#print(X_batch_resized.shape, y_batch_resized_Attn.shape, y_batch_resized_CTC.shape  ,encoder_mask.shape , times, w_max )
def getDataByGenerator(dataset,mode):
    if dataset == 'Syntext':
        df = pd.read_csv('df_train_syn.csv')
        char2int = np.load('char2int_CTC_syn.npy', allow_pickle=True)  #93
        vocab = list(char2int.tolist().keys())
        lenvoc= len(vocab)
        Maxlen = np.load('mx1_syn.npy')

    elif dataset == 'MJsyn':
        df = pd.read_csv('df_MJsyn.csv')
        vocab = np.load('vcMJsyn.npy', allow_pickle=True)  # 62
        vocab = list(vocab.tolist())
        lenvoc =len(vocab)
        Maxlen = np.load('mx1_MJsyn.npy')
    return data_generator(df, batch_size=1, shuffle= cfg.Shuffle, H =cfg.targetHeight, seqDivider=cfg.SeqDivider, num_channel=1, vocab=vocab,
                          padCTC=len(vocab), padTran=0) , Maxlen , lenvoc