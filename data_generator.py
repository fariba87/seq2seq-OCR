import tensorflow as tf
import numpy as np
import cv2
from ConFig.Config import ConfigReader
cfg = ConfigReader()
import pandas as pd
# df = pd.read_csv('df_MJsyn.csv')
# vocab = np.load('vcMJsyn.npy', allow_pickle=True)  # 62
# vocab = list(vocab.tolist())
# dflen = len(df)
# numstep = np.int32(np.divide(dflen , cfg.batchSize))

class data_generator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = cfg.batchSize,
                 shuffle=False,
                 H=cfg.targetHeight,
                 seqDivider =cfg.SeqDivider,
                 num_channel =1,
                 vocab=['a'],#vocab,
                 mode='ctc1',
                 padCTC = 100,#len(vocab),
                 padTran =0):
        self.df = df
        self.batch_size = batch_size
        self.HTraget = H
        self.number_of_samples = len(self.df)
        self.seqDivider = seqDivider
        self.numchannel =num_channel
        self.shuffle =shuffle
        self.vocab=vocab
        self.padCTC = padCTC,#len(self.vocab)
        self.padTran = padTran
        self.mode=mode
        self.voc_CTC= self.vocab
        self.char_CTC =self.voc_CTC
        self.voc_Att = ['SOS']+self.vocab+['EOS']
        self.chars_att=['PAD']+self.voc_Att
        self.chars_Attention = {ch: i for i, ch in enumerate(self.chars_att)}
        self.chars_CTC = {ch: i for i, ch in enumerate(self.char_CTC)}
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return int(len(self.df) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batches = self.df.iloc[indices]#index * self.batch_size:(index + 1) * self.batch_size]  # .iloc[]
        X, yAtt,yCTC, mask , t ,w_max, target_weights= self.__get_data(batches)  # or y1 and y2

        if self.mode =='ctc2':
            #return (x, y, input_length, label_length), y
            return [np.asarray(X, 'float64'),yCTC,np.ones(self.batch_size)*(t), np.ones(self.batch_size)*yCTC.shape[1]] , yCTC  #but default y is len(vocab)
        elif self.mode == 'transformer' :
            return [X, yAtt, mask], yAtt
        elif self.mode =='ctc1':        # if self.CTC:
            return [(X, yCTC), yCTC]
        else:
            return X, yAtt,yCTC, mask , t ,w_max, target_weights
        # if self.Transformer:
        #     return (X, mask,yAtt ) ,
        #return X, yAtt,yCTC , mask , t , w_max  # return a complete batch

    def process_image(self, Hmax, Wmax):  # , H, W):  # there is a problem here! since for scale i got 0
        img = cv2.imread(self.impath, 0)# scale w based on max W in that ds or alllllll?!
        # if img is not None:
        h, w = img.shape[:2]
        # else:
        #     h=0
        #     w=0

        scale = self.HTraget / h
        nw = scale * w
        img = 2.0 *(np.array(cv2.resize(img, (np.int0(nw),self.HTraget))) / 255.)-1
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
            if cv2.imread(self.impath, 0) is not None:
                img, w , t = self.process_image( Hmax_ds[i] , Wmax_ds[i] )

            else:
                img=np.zeros((cfg.targetHeight,3,1))
                w = 3
                t= 3
            times.append(t)
            if w > w_max:
                w_max = w#np.int0(w)
            if len(str(text_list[i])) > max_seq_len:
                max_seq_len = len(str(text_list[i]))  # max_len_per_bacth #text_list_Encoded_CTC

            x_batch.append(img)
        if self.mode =="transformer":
            w_max=800

        X_batch_resized = np.zeros((self.batch_size, self.HTraget, w_max, 1), dtype=np.float32)

        y_batch_resized_CTC  = np.zeros((self.batch_size, max_seq_len), dtype=np.float32)+self.padCTC  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        y_batch_resized_Attn = np.zeros((self.batch_size, max_seq_len+2), dtype=np.float32)+self.padTran  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
        encoder_mask = np.zeros((self.batch_size, np.int32(np.ceil(np.divide(w_max,self.seqDivider)))))
        target_weights = np.zeros((self.batch_size, max_seq_len+2), dtype=np.float32)

        for i in range(self.batch_size):
            X_batch_resized[i, :, :x_batch[i].shape[1], :] = x_batch[i]
            en_Att = np.concatenate((np.expand_dims(self.chars_Attention['SOS'], 0),
                                     np.array([self.chars_Attention[ch] for ch in str(text_list[i])], dtype=np.int32),
                                     np.expand_dims(self.chars_Attention['EOS'], axis=0)), axis=0)

            en_CTC =  np.asarray([self.chars_CTC[ch] for ch in str(text_list[i])], dtype=np.int32)
            y_batch_resized_CTC[i, :len(en_CTC)] = en_CTC
            y_batch_resized_Attn[i, :len(en_Att)] =en_Att
            encoder_mask[i][np.int32(np.divide(x_batch[i].shape[1],self.seqDivider)):] = 1# np.ones_like((1,x_batch[i].shape[1])
            target_weights[i][:len(en_Att) - 1] = 1

        return X_batch_resized, y_batch_resized_Attn, y_batch_resized_CTC  ,encoder_mask , np.array(times), w_max, target_weights


def getDataByGenerator(dataset, mode):
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
    return data_generator(df, batch_size = cfg.batchSize,shuffle = cfg.Shuffle, H=cfg.targetHeight, seqDivider =cfg.SeqDivider,
                         num_channel =1,vocab=vocab, mode=mode, padCTC = len(vocab), padTran =0) , Maxlen , lenvoc, vocab


        #data_generator(df, batch_size=1, shuffle=False, H =cfg.targetHeight, seqDivider=cfg.SeqDivider, num_channel=1, vocab=vocab,
         #                 padCTC=len(vocab), padTran=0) , Maxlen , lenvoc