# remember cuda 1 and 0 is inversed ! mine is 0
import numpy as np
import tensorflow as tf
import pandas as pd

###################################################################################################################
#for sanity check
ctc_label = np.array([ 0,  2, 42, 59, 68, 55, 54, 64, 55, 69, 69,  2], dtype=np.int32)
ctc_label_extend = np.array([[ 0,  2, 42, 59, 68, 55, 54, 64, 55, 69, 69,  2]+[-1]*(25-12)], dtype=np.int32)
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
wres = (2155/1410)*64
wt= int(np.ceil(wres)) #98
SeqDivider=4

import cv2
img = cv2.imread(impath)

h, w ,_= img.shape
ht =64
wnew = np.int32(np.ceil((ht/h)*w))
img = (cv2.resize(img, (wnew, ht))/255.)-0.5

#img= np.expand_dims(img, axis=0)
maxW=600
X = np.zeros((1, ht, maxW, 3))
X[0][:,:img.shape[1] , :]= img
t = np.ceil(img.shape[1] / SeqDivider)
###################################################################################################################

#from transformer_mahdavi import Transformer
#from Transformer_Mahdavi.custom_layers.transformer import Transformer
#from Transformer_Mahdavi.model import Model
#tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
import cv2
#from ConFig.Config import ConfigReader
#ConFig= ConfigReader()
#from dataloader import char2int
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices("GPU")
#char2int_Attn = np.load('char2int_Attn.npy', allow_pickle=True)
#char2int_Attn_array=list(char2int_Attn.item(0).keys())
char2int = 3
vocab= {'a', 'b', 'r'}
chars_sorted = sorted(vocab)
chars_sorted = list(chars_sorted)
chars_Attn = chars_sorted + ["PAD", "SOS", "EOS"]
char_array_Attn = np.array(chars_Attn)
from attn_try import Seq2SeqDynamicModel
from FeatureExtraction import ResNet_backbone

X_batch = np.load('X_batch.npy')#,valgen_Attn.__getitem__(0)[0])
y_batch_Attn = np.load('y_batch_Attn.npy')#,valgen_Attn.__getitem__(0)[1])
y_batch_CTC = np.load('y_batch_CTC.npy')#,valgen_Attn.__getitem__(0)[2])
# i should load times too
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_memory_growth(prhysical_devices[0], True)
#from rectification import rectify
char2int_Attn = np.array([1,2, 3, 4])
char2int_Attn_array = char2int_Attn
#################################################################
dftest = pd.read_csv('/media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/sample_df.csv')
Wmax = max(pd.unique(dftest['Wmax']).tolist())  #2155
Hmax = max(pd.unique(dftest['Hmax']).tolist())  #1410
Htarget = 64
Wtarget = np.ceil((Wmax/Hmax)*Htarget)
seq_divider = 4
seq_len = int(np.ceil(Wtarget  / seq_divider))
#times = tf.cast(tf.tile(tf.expand_dims(seq_len, 0), [batchSize]), dtype=tf.int32)
Lenmax = maxlen =27 # for this dataset
###################################################################
def Adj_mat(n):
    beta = 0.5
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i:] = A[i:, i] = np.arange(n - i)
    A = tf.convert_to_tensor(A)
    A_adj = 1 - tf.math.sigmoid(A+beta)
    return A_adj
max_seq_length=300
Adj_mat_list = list(map(Adj_mat, range(max_seq_length)))
#np.save('Adj_mat_list.npy', Adj_mat_list)
class CTCLayer(tf.keras.layers.Layer):
    # i wrote this class , (so that in model compilation, there is no need to define loss)
    # but at last is was not used
    def __init__(self, name=None):#,  cfg: ConfigReader):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
        '''gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = tf.fill((batch_len,), model_output.shape[1])
        ctc_loss = tf.nn.ctc_loss(
            gt, model_output, seq_len, input_length, logits_time_major=False, blank_index=len(self.vocab)
        )'''
    def build_target(self):
        ytrue = self.y_true
        list_target = [tf.gather(ytrue ,t) for t in range(tf.shape(ytrue)[0])]
        idx=tf.where(tf.not_equal(ytrue ,-1))
        gt=np.zeros((tf.shape(ytrue)[0], 2))
        seq_length = len(idx)
        return gt , seq_length
    def call(self, y_true, y_pred):
        # if ytrue is raw CTC label call build_target
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64") #ok
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") #ok
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        self.y_true =y_true
        process_ytrue=False
        if process_ytrue:
            gt , seq_len =self.build_target()
            lossctc = tf.nn.ctc_loss(
                gt, y_pred, seq_len, input_length, logits_time_major=False, blank_index=len(self.vocab)
                # vocab should be defined in init
            )
            seq_len = tf.cast(seq_len, tf.int32)
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        # i think i should use the correct length for input and label
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")# i should change these length
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # At test time, just return the computed predictions
        return y_pred #, lossctc
class GraphConvolutionLayer(tf.keras.layers.Layer):
    #input X :[BS, T, Q]
    # A_S :[T,T]
    # A_D :[T,T]
    # H : [T,Q]
    # W : [Q, N]
    # output: BS*T*N
    def __init__(self, units, activation=tf.identity): # A
        super(GraphConvolutionLayer, self).__init__()

        self.activation = activation
        self.units = units  # what is unites? =N
        #self.A = A it can also be defined for constant sequence length for each batch

    def build(self, input_shape):  # what is input shape?
        self.W = self.add_weight(
            shape=(input_shape[2], self.units), #unit is N in defination of W , input_shape [BS, T,Q]
            dtype=self.dtype,
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(0.01))


    def sim_mat(self):
        h = self.Hmatrix  #[None, 40 ,1024[

        n = h.shape.as_list()[-1]
        h = tf.keras.layers.Dense(n)(h)
        L2Norm = tf.sqrt(tf.reduce_sum(tf.pow(h, 2), axis=-1))
        h1 = h / tf.tile(tf.expand_dims(L2Norm, axis=-1), [1, 1, n])#sh[-1]])
        A_S = tf.cast(tf.matmul(h1, tf.transpose(h, [0, 2, 1])),tf.float32)
        A_S = tf.keras.layers.Dot(axes=2, normalize=True)([h, h])

        #A_S = tf.matmul(H, tf.transpose(H),) / (tf.pow(tf.math.l2_normalize(H), 2))
        #A_S = tf.matmul(tf.transpose(h, perm=[0, 2, 1]), h) / (tf.pow(tf.tf.math.l2_normalize(h), 2))
        #self.A_S = A_S
        return A_S   # [none , 40, 40]

    def call(self, X, GCNin):  #in calling :both X and resnet.output
        self.X = X  #[BS, T, Q]
        self.Hmatrix = GCNin#[0,:,:]
        n = GCNin.shape.as_list()[-2]  #40
        bs = tf.shape(GCNin)[0]

        #bs=32
        A_D = tf.tile(tf.expand_dims(Adj_mat_list[n],axis=0),[bs,1,1])
        A_D = tf.cast(A_D, dtype= tf.float32) #how can i resize?
        print(A_D.shape)
        #A_D = tf.keras.layers.Lambda(lambda: Adj_mat_list[40])()


        # X = tf.nn.dropout(X, self.rate)
        #self.A = tf.matmul (self.sim_mat()*self.Adj_mat())
        #n = self.X.get_shape().as_list()[-2]
      #  self.A = tf.matmul(self.sim_mat(),A_D)#dj_mat_list[n])#self.A_adj)
        self.A  =self.sim_mat()@A_D
        X = self.A @ self.X @ self.W
       # h = tf.matmul(h @ h.T) / tf.math.l2_normalize(h @ h.T)
        return X #self.activation(X)
class GCTC(object):
    '''
    within this class : from single input(text images).
    two model(CTC-Attn-Transformer) are created with different loss
    if data generator :
           yield(X_batch, {'output1': y1_batch, 'output2': y2_batch} ))
    and in one model
    model.compile(optimizer='adam', loss={'output1': 'mean_squared_error', 'output2': 'mean_squared_error'})
        '''
    def __init__(self, feature_extraction='ResNet', trainable=True):#,  cfg: ConfigReader):
        #self.config = cfg.modelConfig
        # datagenerator output : x_batch y_ctc_batch , y_Attn_batch
        self.input = tf.keras.layers.Input(shape=(64,600,3))#, (64, None, 3)#batch_size=32)#224, 224, 3))
        self.labels_CTC = tf.keras.layers.Input(shape =(25)) # (None)#should be None  # 17
        self.labels_Attn = tf.keras.layers.Input(shape = (19))  #(None)  # 19
        self.times = tf.keras.layers.Input(shape =())


        # labels_CTC , labels_Attn [Bs*maxlen in that batch]

        self.model_Attn = None
        self.model_CTC  = None
        self.model_transformer = None
        #self.output_CTC  = y_ctc  # how to consider?
        #self.output_Attn  =  y_Attn
        self.STN_apply=False
        self.model_CTC_ok=None


        self.trainable =trainable
        self.ctc_prediction_len = 90
        if feature_extraction == 'ResNet':
            self.featuremap =self.ResNet_backbone
        else:
            self.featuremap =self.Mobilenet_backbone
    def STN(self, x):
        # it is not needed since we rectify manually
        # and test samples are originally rectified
        def localization_net():
            pass
        def grid_generator():
            pass
        # or call rectify method from rectification module
        #return normalized_image
    def Mobilenet_backbone(self , inp): #instead of resnet

        def mobilnet_block(x, filters, strides):
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x

  #      input = self.input#Input(shape=(224, 224, 3)) # # TODO: change
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=(None, None, 64,1))(inp)#(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # main part of the model
        x = mobilnet_block(x, filters=64, strides=1)
        x = mobilnet_block(x, filters=128, strides=2)
        x = mobilnet_block(x, filters=128, strides=1)
        x = mobilnet_block(x, filters=256, strides=2)
        x = mobilnet_block(x, filters=256, strides=1)
        x = mobilnet_block(x, filters=512, strides=2)
        for _ in range(5):
            x = mobilnet_block(x, filters=512, strides=1)
        x = mobilnet_block(x, filters=1024, strides=2)
        x = mobilnet_block(x, filters=1024, strides=1)
        x = tf.keras.layers.AveragePooling2D(pool_size=7, strides=1, data_format='channels_first')(x) #(None, 7, 1, 1018)
        x=tf.keras.layers.Lambda (lambda y: tf.keras.backend.squeeze(y, 2))(x)
        #output = Dense(units=1000, activation='softmax')(x) # TODO: delete it
        mobilenet_model = tf.keras.models.Model(inputs=input, outputs = x)# output)
        mobilenet_model.summary()
        return x, mobilenet_model  # x is our feature map that should be resized to feature vector seq
    def ResNet_backbone(self,Xin):

        def residual_block1(res_n_ch,X):
            a, b, c = res_n_ch
            '''model_res_block = tf.keras.models.Sequential([tf.keras.layers.Conv2D(a, (1, 1)),
                                                          tf.keras.layers.Conv2D(b, (3, 3)),
                                                          tf.keras.layers.Conv2D(c, (1, 1))])'''
            shortcut=X
            X1 = tf.keras.layers.Conv2D(a, (1, 1), padding='valid')(X)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X1 = tf.keras.layers.Activation('relu')(X1)

            X1 = tf.keras.layers.Conv2D(b, (3, 3), padding='same')(X1)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X1 = tf.keras.layers.Activation('relu')(X1)

            X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X_shortcut = tf.keras.layers.Conv2D(filters = c, kernel_size = (1, 1),  padding = 'valid')(shortcut)
            X_shortcut = tf.keras.layers.BatchNormalization(axis = 3)(X_shortcut)
            X1 = tf.keras.layers.Add()([X_shortcut, X1])
            Xo = tf.keras.layers.Activation('relu')(X1)
            return  Xo
        def identity_block(res_n_ch,X):
          #  conv_name_base = 'res' + str(stage) + block + '_branch'
          #  bn_name_base = 'bn' + str(stage) + block + '_branch'
            a, b, c = res_n_ch
            shortcut = X
            X1 = tf.keras.layers.Conv2D(a, (1, 1), padding='valid')(X)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X1 = tf.keras.layers.Activation('relu')(X1)

            X1 = tf.keras.layers.Conv2D(b, (3, 3), padding='same')(X1)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X1 = tf.keras.layers.Activation('relu')(X1)

            X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='valid')(X1)
            X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
            X = tf.keras.layers.Add()([shortcut, X])  #?
            X = tf.keras.layers.Activation('relu')(X)
            return X
        res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
      #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
        Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
        X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(Xi) #
        X = tf.keras.layers.BatchNormalization(axis=3)(X) #
        X= tf.keras.layers.ReLU()(X) #
        X = tf.keras.layers.MaxPool2D((3, 3), (2, 2))(X) #

        X = residual_block1(res_n_ch[0], X)
        X = identity_block(res_n_ch[0], X)
        X = identity_block(res_n_ch[0], X)

        X = residual_block1(res_n_ch[1], X)
        X = identity_block(res_n_ch[1], X)
        X = identity_block(res_n_ch[1], X)
        X = identity_block(res_n_ch[1], X)
        X = tf.keras.layers.MaxPool2D((2, 1), (2, 1))(X)

        X = residual_block1(res_n_ch[2], X)
        X = identity_block(res_n_ch[2], X)
        X = identity_block(res_n_ch[2], X)
        X = tf.keras.layers.MaxPool2D((2, 1), (2, 1))(X)

        X = residual_block1(res_n_ch[3], X)
        X = identity_block(res_n_ch[3], X)
        X = identity_block(res_n_ch[3], X)  # 3, 38 ,256
        X = tf.keras.layers.ZeroPadding2D((1, 1))(X)
        Xo = tf.keras.layers.AvgPool2D((4, 1), (1, 1))(X)  # 2,40,256
        Xo = tf.keras.layers.AvgPool2D((2, 1), (1, 1))(Xo)  # 2,40,256

        Xo = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 1))(Xo)  #40 ,256 # or use feature_map_to_featurevec function
        resnet_model = tf.keras.models.Model(inputs=Xin, outputs=Xo)
        resnet_model.summary()
        return Xo , resnet_model
    #no need to this function if squeeze is applied beforehand
    def featuremap_to_featurevec(self,x) :#, resnet = True):
        #in the case we didnt achieve to Dim= 1 (so not squeezable) we do this way
        featuremap = x
        [bs, h,w,c]= featuremap.shape# feature_map.shape
        # feature maps shape [BS*H*W*C]
        feature_vec = np.reshape(featuremap, (bs, w, c * h))
        return feature_vec
    def attential_guidance(self, X, training=True):
        # input :feature vector  [None ,40,256 ]
        # output char sequence
        # based on GRU
        #tf.keras.layers.GRU
        #tf.keras.layers.GRU
        based_on_keras_attention =None
        if based_on_keras_attention:
            from attnkeras import attention_with_keras
            encoder_in = X
            decoder_in = self.labels_Attn
            decoder_out= self.labels_Attn
            #encoder_in, decoder_in, decoder_out = train_dataset
             #= attention_with_keras(checkpoint_dir, encoder_in, decoder_in, decoder_out, vocab_size, maxlen, batch_size=)



        #[Bs , W , C]=X.get_shape().asattn_num_hidden_list()
        [Bs , W , C] = tf.shape(X)
        # W = 10
        #Bs = 32
        #Bs= tf.shape(self.labels_Attn)[0]
        # C = 500
        max_pred_length = 25
        #perm_conv_output = tf.zeros((W, Bs, C))
        decoder_size = max_pred_length + 3

       # decoder_inputs = [[1.] * Bs] * decoder_size

        #target_weights = [ [1.]* Bs] * (decoder_size - 1) + [[0.] * Bs]
        num_classes = 80  # len(vocab)+2
        encoder_size = W

        target_embedding_size = 10
        attn_num_layers = 2
        attn_num_hidden = 128
        is_training = True
        #########
        labels = self.labels_Attn
        [Bs, max_seq_batch] =self.labels_Attn.get_shape()#.shape
        decoder_size = self.labels_Attn.get_shape()[1]#  #since i consider SOS and EOS in label encoding

        encoder_size =tf.cast(tf.math.ceil(tf.divide(encoder_size,4)), tf.int32)
        buckets = [(encoder_size, decoder_size)]
        #temp = tf.zeros((Bs, (max_pred_length + 2)-max_seq_batch)) - 1

        templast=tf.zeros((tf.shape(self.labels_Attn)[0],1))
        decoder_inputs = tf.concat([labels,  templast ] ,axis=1)
        #decoder_inputs =tf.where(self.labels_Attn , self.labels_Attn, 0.)
        #decoder_inputs[:, :max_seq_batch] = self.labels_Attn
        decoder_inputs1 = tf.where(decoder_inputs==-1, 0., decoder_inputs)
        target_weights1 = tf.where(decoder_inputs1==0., 0. , 1.)
        # should change to list
        decoder_inputs_list =[]
        target_weights_list =[]
        for i in range(decoder_size+1):
            decoder_inputs_list.append(decoder_inputs1[:,i])
            target_weights_list.append(target_weights1[:,i])

        #######
        perm_X = tf.reshape(X, (W,-1, C))#-1, W, C))  #[104,None,2048]
        attention_decoder_model = Seq2SeqDynamicModel(
            encoder_inputs_tensor=perm_X, #conv_output,
            decoder_inputs=decoder_inputs_list,
            target_weights=target_weights_list,
            target_vocab_size=num_classes,
            buckets=buckets,
            target_embedding_size=target_embedding_size,
            attn_num_layers=attn_num_layers,
            attn_num_hidden=attn_num_hidden,
            forward_only=not (is_training),
        )
        pred1 = attention_decoder_model.output

        num_feed = []
        prb_feed = []
        #attention_decoder_model.output [19,None, 80
        for line in range(len(attention_decoder_model.output)):
            guess = tf.argmax(attention_decoder_model.output[line], axis=1)  #none
            proba = tf.reduce_max(
                tf.nn.softmax(attention_decoder_model.output[line]), axis=1)
            num_feed.append(guess) #19 ta none
            prb_feed.append(proba) #19 ta none

        '''

        tr_all=[]
        for i in range(len(num_feed)-1, -1,-1):
            tr = tf.cond(tf.equal(num_feed[i],78), lambda: '', lambda: char_array_Attn[num_feed[i]])## tf.compat.v1.map_fn(lambda m:
            tr_all.append(tr)
           #, num_feed[i], dtype=tf.string)
        trans_output =tr_all

        @tf.function
        def helper_fn(a):
            A = tf.cond(tf.equal(a, 78), lambda: '', lambda: char_array_Attn[a])  #
            #
        tr_all=[]
        for i in range(len(num_feed)-1, -1,-1):
            tr = helper_fn(num_feed[i])#tf.cond(tf.equal(num_feed[i],78), lambda: '', lambda: char_array_Attn[num_feed[i]])## tf.compat.v1.map_fn(lambda m:
            tr_all.append(tr)'''


        table = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="",
            checkpoint=True,
        )

        insert = table.insert(
            tf.constant(list(range(10)), dtype=tf.int64), #self.config.num_classes
            tf.constant(['r', 't', 'f','w','m','n','q','u','a','x']),) #self.config.CHARMAP

        charset = ['r', 't', 'f', 'w', 'm', 'n', 'q', 'u', 'a', 'x','3','5','6','b','c','7','l','k','u', 'EOS']
        char2int_Attn = {ch:i  for i, ch in enumerate(charset)}
        char_array_Attn=tf.convert_to_tensor(np.array(charset))
        char_array_Attn = np.array(charset)
        # Join the predictions into a single output string.
        trans_output = tf.transpose(num_feed)  #none,19

        solution1=solution0=False

        if solution0:
            class ArithmeticLayer(tf.keras.layers.Layer):
                # u = number of units

                def __init__(self, name=None, regularizer=None, unit_types=['id', 'sin', 'cos']):
                    self.regularizer = regularizer
                    super().__init__(name=name)
                    self.u_types = tf.constant(unit_types)
                    self.u_shape = tf.shape(self.u_types)


                def call(self, inputs):
                    dense_output_nodes = inputs
                    d_shape = tf.shape(dense_output_nodes)
                    i = tf.constant(0)
                    c = lambda i, d: tf.less(i, self.u_shape[0])

                    def b(i, d):
                        k = tf.foldr(
                            lambda a, x: tf.cond(
                                tf.equal(x, 10),  # self.config.EOS_ID), # EOS_ID = akharin bood
                                lambda: '',
                                lambda: '3'#tf.keras.backend.cast(char_array_Attn[x] + a, tf.string)  # table.lookup(x) + a  #
                            ),
                            d,
                            initializer='',

                        )
                        # d = tf.cond(unit_types[i] == 'sin',
                        #             lambda: tf.tensor_scatter_nd_update(d, tf.stack(
                        #                 [tf.range(d_shape[0]), tf.repeat([i], d_shape[0])], axis=1),
                        #                                                 tf.math.sin(d[:, i])),
                        #             lambda: d)
                        # d = tf.cond(unit_types[i] == 'cos',
                        #             lambda: tf.tensor_scatter_nd_update(d, tf.stack(
                        #                 [tf.range(d_shape[0]), tf.repeat([i], d_shape[0])], axis=1),
                        #                                                 tf.math.cos(d[:, i])),
                        #             lambda: d)
                        return k

                    dense_output_nodes = tf.while_loop(c, b, loop_vars=[i, dense_output_nodes])
                    return dense_output_nodes

        if solution1:

            max_loop = tf.shape(trans_output)[0]

            def should_continue(t, *args):
                return t < max_loop

            def iteration(t, m, outputs_):
                cur = tf.gather(trans_output, t)
                k= tf.foldr(
                            lambda a, x: tf.cond(
                                tf.equal(x, 10),#self.config.EOS_ID), # EOS_ID = akharin bood
                                lambda: '',
                                lambda: char_array_Attn[x]+a#table.lookup(x) + a  #
                            ),
                            cur,
                            initializer='',

                        )

                #m = m * 0.5 + cur * 0.5
                #outputs_ = outputs_.write(t, m)
                return k#t + 1, m, outputs_

            #tf.compat.v1.disable_eager_execution()
            i0=tf.range(max_loop)
            m0= [tf.gather(trans_output,i) for i in tf.range(max_loop)]

            outputs = tf.while_loop(cond= lambda i ,m: tf.less(i, max_loop),
                                     body= lambda i, m : tf.foldr(
                            lambda a, x: tf.cond(
                                tf.equal(x, 10),#self.config.EOS_ID), # EOS_ID = akharin bood
                                lambda: '',
                                lambda: '3'#char_array_Attn[x]+a#table.lookup(x) + a  #
                            ),
                            m,
                            initializer='',

                        ), loop_vars=[i0, m0])
                                    #trans_output,  # loop_vars , loop_vars=[i0, m0],
                                    #shape_invariants=None)  #



            outputs = tf.while_loop(should_continue, # cond: cond = lambda i, result: tf.less(i, max_loop)
                                    iteration,#body
                                    trans_output, #loop_vars , loop_vars=[i0, m0],
                                    shape_invariants= None)#tf.TensorShape([0,trans_output.get_shape()[1]]))#tf.shape(trans_output)[1]])) #
                                          #[initial_t, initial_m, initial_outputs])
            outputs = outputs.stack()
        solution2=True
        if solution2:

            class Linear1(tf.keras.layers.Layer):
                def __init__(self, units=32, input_dim=32):
                    super(Linear1, self).__init__()


                def call(self, inputs):
                    A = tf.map_fn(
                        lambda m: tf.compat.v1.foldr(  # like last in first out
                            lambda a, x: tf.compat.v1.cond(
                                tf.compat.v1.equal(x, 3),  # EOS_ID = akharin bood
                                lambda: '',
                                lambda: '2'# tf.cast(char2int_Attn_array[x]+a , tf.string)#char_array_Attn[x] + a  # table.lookup(x) + a  #
                            ),
                            m,
                            initializer=''
                        ),
                        inputs,
                        dtype=tf.string
                    )
                # prediction = tf.cond(
                #     tf.equal(tf.shape(A)[0], 1),
                #     lambda: trans_output[0],
                #     lambda: trans_output,
                # )
                    return A#prediction
            trans_output =Linear1()(trans_output)

        trans_outprb = tf.compat.v1.transpose(prb_feed)
        solution3=False
        if solution3:
            def map_func1(m):
                return  tf.foldr(#like last in first out
                    lambda a, x: tf.cond(
                        tf.equal(x, 10),#self.config.EOS_ID), # EOS_ID = akharin bood
                        lambda: '',
                        lambda: '3'#char_array_Attn[x]+a#table.lookup(x) + a  #
                    ),
                    m,
                    initializer='',

                )

            trans_output = tf.keras.layers.Lambda(map_func1)(trans_output)

        trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))

        class Linear2(tf.keras.layers.Layer):
            def __init__(self, units=32, input_dim=32):
                super(Linear2, self).__init__()


            def call(self, inputs):
                return  tf.map_fn(
                lambda m: tf.foldr(
                    lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                    m,
                    initializer=tf.cast(1, tf.float64)
                ),
                inputs,
                dtype=tf.float64
            )

        # def map_func2(m):
        #     return tf.foldr(
        #         lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
        #         m,
        #         initializer=tf.cast(1, tf.float64)
        #     )#,

        #trans_outprb =tf.keras.layers.Lambda(map_func2)(trans_outprb)
        trans_outprb =Linear2()(trans_outprb) #none
        loss = attention_decoder_model.loss
        self.loss=loss





        '''trans_outprb = tf.compat.v1.map_fn(
            lambda m: tf.foldr(
                lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                m,
                initializer=tf.cast(1, tf.float64)
            ),
            trans_outprb,
            dtype=tf.float64
        )


        # tf.compat.v1.disable_eager_execution()
       #  import tensorflow.compat.v1 as tff
       #  tff.disable_v2_behavior
       #  [bs, seqlen]=tf.shape(trans_output)
       #  for i in range(seqlen-1,-1, -1)
       #      tf.cond(tf.equal(trans_output[i], 10.))
    #  @tf.function





        trans_output = tf.map_fn(
            lambda m: tf.compat.v1.foldr(#like last in first out
                lambda a, x: tf.compat.v1.cond(
                    tf.compat.v1.equal(x, self.config.EOS_ID), # EOS_ID = akharin bood
                    lambda: '',
                    lambda: char_array_Attn[x]+a #table.lookup(x) + a  #
                ),
                m,
                initializer=''
            ),
            trans_output,
            dtype=tf.string
        )'''

        # Calculate the total probability of the output string.


        prediction =trans_output
        probability=trans_outprb
        #
        # prediction = tf.cond(
        #     tf.equal(tf.shape(trans_output)[0], 1),
        #     lambda: trans_output[0],
        #     lambda: trans_output,
        # )
        # probability = tf.cond(
        #     tf.equal(tf.shape(trans_outprb)[0], 1),
        #     lambda: trans_outprb[0],
        #     lambda: trans_outprb,
        # )

        if training:
            # Join the predictions into a single ground string.
            trans_ground = tf.cast(tf.transpose(decoder_inputs), tf.int64)
            trans_ground =Linear1()(trans_ground)
            # trans_ground = tf.map_fn(
            #     lambda m: tf.foldr(
            #         lambda a, x: tf.cond(
            #             tf.equal(x, self.config.EOS_ID),  # EOS :akharin bood
            #             lambda: '',
            #             lambda: char_array_Attn[x]+a #table.lookup(x) + a  # pylint: disable=undefined-variable
            #         ),
            #         m,
            #         initializer=''
            #     ),
            #     trans_ground,
            #     dtype=tf.string
            # )
            # ground = tf.cond(
            #     tf.equal(tf.shape(trans_ground)[0], 1),
            #     lambda: trans_ground[0],
            #     lambda: trans_ground,
            # )

            ground=trans_ground

        #self.prediction = tf.identity(self.prediction, name='prediction')
        #self.probability = tf.identity(self.probability, name='probability')

        #attention (defined only for training )
        return pred1
    ################################################# the first three part is trained by CE loss
    def GCN(self,x, GCNin): # h is output from Mobilenet(squeezed)
        #adjacency_matrix = similarty_matrix*distance_matrix
        #output [bs, w, , c*h']
        GCNlayer_output = GraphConvolutionLayer(units=400)(x, GCNin) # (A_S * A_D) * H * W_g  # W_g :weight matrix  # H: h 1:T
        return GCNlayer_output #as LSTM_input
    def BiLSTM(self,X):
        # I tried the way with tf.nn.bidirectional from TF1 (that has an argument for sequence length) but is didnt work in TF2
        # so I am using tf.keras.layers.Bidirectional+ Masking instead.

        # bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        # fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        # fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(128)
        # bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(128)
        # outputs, outputs_states = tf.compat.v1.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, GCNout, sequence_length=self.times, dtype=tf.float32)
        # outputs = tf.concat([outputs[0], outputs[1]], 2)
        # outputs = tf.reshape(outputs, [-1, 128 * 2])

        #     fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(128)
        #     bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(128)
        #     lstm_outputs_fw, _ = tf.compat.v1.nn.dynamic_rnn(
        #         fw_cell,
        #         GCNout,
        #         sequence_length=self.times,
        #         dtype=tf.float32)
        #     GCNout_reverse = tf.reverse_sequence(GCNout, seq_lengths=self.seqlen, seq_dim=1)
        #     tmp, _ = tf.nn.dynamic_rnn(
        #         bw_cell,
        #         GCNout_reverse,
        #         sequence_length=self.seqlen,
        #         dtype=tf.float32)
        #     lstm_outputs_bw = tf.reverse_sequence(tmp, seq_lengths=self.seqlen, seq_dim=1)
        #     lstm_outputs = tf.add(lstm_outputs_fw, lstm_outputs_bw, name="lstm_outputs")
        max_len = tf.shape(X)[1]
        mask = tf.expand_dims(tf.sequence_mask(self.times, maxlen=max_len, dtype=tf.float32, name=None), -1) #tf.reduce_max(self.times)
        # mask should be of size [None, seqlen, 1]
#        mask = tf.expand_dims(tf.sequence_mask([6, 10], maxlen=15, dtype=tf.float32, name=None), -1)
        inp= X*mask
        X1 = tf.keras.layers.Masking(mask_value=0.0)(inp)
        X1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(X1)
        #BiLSTM.add(tf.keras.layers.TimeDistributed(Dense(1, activation='sigmoid')))
        return X1
    def classifier1(self,for_Attn):
        # if for_Attn:
        #     n_class = self.vocab_size+2# go+eos (2) #
        # else:
        #     n_class = self.vocab_size+1#blank (1)
        n_class =78
        classifier = tf.keras.layers.Dense(n_class, activation='softmax', name="dense")(self.seq) #BS *Timesteps*numclasses
        return classifier
    def feature_extraction(self,x):
        if self.STN_apply:
            x= self.STN(x)
        x =self.featuremap(x)
#       x =self.featuremap_to_featurevec(self, x)
        return x
    def forward_transformer(self):
        Input = self.input #[None, 64,411,3]
        #X = self.featurmap(Input)
        layer_out, resnet_model = self.feature_extraction(Input)# self.input)#image)  #
        self.resnet_model = resnet_model
        #self.transformer = 1

        #model_transformer = Model(hparams=1)
        #model_transformer.input_embedding_layer = resnet_model
        input_embeddings = resnet_model
        tar_real = self.labels_CTC[:, 1:] # np.copy()
        batch_target = self.labels_CTC[:, :-1]
        batch_target[batch_target == char2int_Attn['EOS']]= 0#self.dataLoader.char_to_idx[self.dataLoader.hparams.end_symbol]] = 0
        input_embeddings = layer_out  #[none, 104,2048]
        batch_target = self.labels_Attn
        BS = tf.shape(self.labels_Attn)[0]
        maxW_enc = tf.shape(self.input)[2]
        encoderInputMask = tf.zeros(shape=[BS, tf.cast(tf.math.ceil(tf.divide(maxW_enc, 16)), tf.int32)])
        num_layers = 2
        d_model = 512
        num_heads = 8
        dff = 2048
        dropout_rate=0.1

        self.transformer = Transformer(num_layers = num_layers,
                                       d_model = d_model,
                                       num_heads = num_heads,
                                       dff=dff,
                                       target_vocab_size=len(chars_sorted)+1,#self.dataLoader.vocab) + 1, # for PAD add " + 1"
                                       maxSourceLength=self.input.get_shape()[2],#self.dataLoader.maxSeqLen,
                                       maxTargetLength=self.labels_Attn.get_shape()[1] - 1,
                                       rate=dropout_rate)#maxTargetLength=self.dataLoader.targetMaxLen - 1,
        out = self.transformer([input_embeddings, batch_target, encoderInputMask], True)

        predictions, attention_weights= model_transformer.transformer([input_embeddings,#[none, 104,2048]
                                                                       batch_target,  #none,19
                                                                       encoderInputMask], True)  #[none,411]
        #model_transformer.



        model_tr = tf.keras.models.Model(inputs = [self.input ,self.labels_CTC] , outputs = predictions)
        self.model_transformer = model_tr
        def loss_function_tr(self, real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        model_tr.compile(loss= loss_function_tr , optimizer=tf.keras.optimizers.Adam)
        #model_tr.fit([X_batch, y_batch_CTC],y_batch_CTC)
    def forward_Attn(self):#, image):
        Input = self.input
        #X = self.featurmap(Input)
        layer_out, resnet_model = self.feature_extraction(Input)# self.input)#image)  #
        self.resnet_model =resnet_model
     #   out1=self.attential_guidance(layer_out)
        pred1=self.attential_guidance(layer_out)

     #   pred1=self.classifier1(out1, for_Attn=True)
 #      loss = tf.keras.losses.sparse_categorical_crossentropy()
  #     loss = tf.keras.losses.categorical_crossentropy()
        model1=tf.keras.models.Model(inputs = [self.input,self.labels_Attn],outputs=pred1)
        self.model_Attn =model1
        #model1.compile(loss =self.loss , optimizer=tf.keras.optimizers.Adam())
        # gt : go + char+ eos +pad
        return self.loss
    def forward_CTC(self):#, image):
        feat , resnet_model = self.feature_extraction(self.input)
        GCNin = resnet_model.output  #but this is the same as xo
#        resnet_model.load_weights('from check point') #?
        for layer in resnet_model.layers:
            layer.trainable=True

       # GCNout=self.GCN(feat, GCNin)
      #  seq = self.BiLSTM(GCNin)
       # seq=self.BiLSTM(GCNout)
       # self.seq=seq
        self.seq =GCNin
        pred2=self.classifier1(for_Attn=False)
        self.pred2=pred2 #(None,104,81)
        # paste here
        if self.trainable:
            # i should bring the lable here and variable input length
            labels = self.labels_CTC #none,17
            Sh1 = tf.shape(self.labels_CTC)[0]
#            self.times = tf.cast(tf.tile(tf.expand_dims(seq_len, 0), [Sh1]), dtype=tf.int32)
            Sh2 = self.labels_CTC.get_shape()[1]
            # maxlen = tf.shape(labels)[1]  #17
            # ind=tf.cast(tf.where(tf.equal(labels, -1)),tf.int32)
            # seq_len = tf.cast(tf.gather_nd(
            #     indices=ind,
            #     params=labels), tf.int32)
            # [max_time, batch_size, num_classes].
            self.pred2_trans = tf.transpose(self.pred2, [1, 0, 2])
            seq_lengths = tf.fill([Sh1], tf.shape(self.pred2_trans)[0],
                                  name="seq_lengths")
        #    seq_lengths = tf.fill([Sh1], tf.shape(self.labels_CTC)[1],
                                  #name="seq_lengths")

            decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=self.pred2_trans,
                                                               sequence_length=tf.cast(seq_lengths , tf.int32))#tf.tile(tf.expand_dims(Sh2, axis=0),[Sh1])) #tf.cast(self.times, tf.int32))#tf.tile(tf.expand_dims(Sh2, axis=0),[Sh1]))  # ,
            #decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(self.pred2_trans, tf.cast(self.times, tf.int32), merge_repeated=False, top_paths=30)
            #  blank_index = tf.shape(self.pred2_trans)[-1])  # I already add +1 to classifier dense layer
            #decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(self.pred2_trans, seq_lengths, merge_repeated=False, top_paths=30)
            decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(self.pred2_trans,
                                                                        tf.cast(self.times, tf.int32),
                                                                        merge_repeated=False,
                                                                        top_paths=30)

            '''
            inputs	: [max_time, batch_size, num_classes]. 
            sequence_length	1-D int32 vector containing sequence lengths, having size [batch_size].
            merge_repeated	Boolean. Default: True.
            blank_index	(Optional). Default: num_classes - 1.
            Define the class index to use for the blank label. 
            Negative values will start from num_classes, ie, -1 will reproduce 
            the ctc_greedy_decoder behavior of using num_classes - 1 for the blank symbol, which corresponds to the defaul
            neg_sum_logits : [None,1] '''
            res_lab = tf.compat.v1.sparse_to_dense(tf.cast(decoded[0].indices, tf.int32),  #none,90
                                                   tf.stack([tf.shape(self.labels_CTC)[0],
                                                             self.ctc_prediction_len]), decoded[0].values,
                                                   default_value=-1)

            mymodelctc = tf.keras.models.Model(inputs=[self.input, self.labels_CTC , self.times], outputs=self.pred2)#_trans)  # pred2)

            self.model_CTC_ok = mymodelctc
           # acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), self.targets))

            # sp_input = tf.SparseTensor(
            #     dense_shape=a.get_shape(),#labels.get_shape(),
            #     values=a,#labels,
            #     indices=ind)
            #ogit_length = tf.tile(tf.shape(pred2)[1], [tf.shape(labels)[0]]
            ###################################################################
                # output = CTCLayer(name="ctc_loss")(labels, pred2)#softmax_output)
                # def custom_loss(y_true, y_pred):
                #     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")  # ok
                #     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")  # ok
                #     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
                #     self.y_true = y_true
                #     process_ytrue = False
                #     if process_ytrue:
                #         gt, seq_len = self.build_target()
                #         lossctc = tf.nn.ctc_loss(
                #             gt, y_pred, seq_len, input_length, logits_time_major=False, blank_index=81)#len(self.vocab)
                #             # vocab should be defined in init
                #         #)
                #         seq_len = tf.cast(seq_len, tf.int32)
                #     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
                #     # i think i should use the correct length for input and label
                #     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # i should change these length
                #     loss_fn = tf.keras.backend.ctc_batch_cost
                #     loss = loss_fn(y_true, y_pred, input_length, label_length)
                #     return loss
                # self.custom_loss =custom_loss
            #self.losses_CTC = self.model_CTC.losses
            #self.lossCTC = GCTC().custom_loss
            #self.lossCTC = model2.losses
        else :
            model_output=2
            gt=3
            seq_len=4
            input_length=5
            vocab=2
            '''
            greedy decoder input
             input= [max_time x batch_size x num_classes]
             seqlength = [batch_size]
            output : 
            (decoded, neg_sum_logits)
            
            decoded.indices: Indices matrix (total_decoded_outputs x 2). The rows store: [batch, time].
            decoded.values: Values vector, size (total_decoded_outputs). The vector stores the decoded classes.
            decoded.shape: Shape vector, size (2). The shape values are: [batch_size, max_decoded_length]
            
            neg_sum_logits: A float matrix (batch_size x 1) containing, for the sequence found, the negative of the sum of the greatest logit at each timeframe 
            '''
            output = tf.nn.ctc_greedy_decoder(pred2, seq_lens=10, blank_index =81)#len(self.vocab_size))
            input_length = 2
            # pred2 : containing the prediction, or output of the softmax
            # decoder outout : returns a list of one element that contains the decoded sequence.
            output = tf.keras.backend.ctc_decode(self.pred2, input_length=input_length,
                                                 greedy=True)  # ,beam_width=100,top_paths=1)

            '''_decoded = tf.sparse.concat(
                1,
                [tf.sparse.expand_dims(dec, axis=1) for dec in _decoded],
                expand_nonconcat_dims=True,
            )  # dim : batchsize x beamwidth x actual_max_len_predictions
            out_idxs = tf.sparse.to_dense(_decoded, default_value=len(self.vocab))'''
            #
            self.pred2 = pred2
            #output = CTC_decoder()

            '''outputs = tf.nn.ctc_greedy_decoder(logits,seq_lens,blank_index=1)'''
            # or decode by CTC_decoder function
    def loss_CTC_calc(self):#, ytrue, ypred):
        # This loss from the class was not used at last. I defined loss output of GCTC class
        ypred = self.pred2
        labels = self.labels_CTC
        times = self.times
        BS = tf.shape(self.labels_CTC)[0]
        class targetbuild(tf.keras.layers.Layer):
            def __init_(self):
                super(targetbuild, self).__init__()
            def call(self, labels):
                idx = tf.where(tf.not_equal(labels, -1))
                targets = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.cast(tf.shape(labels), tf.int64))
                return targets
        self.targets = targetbuild()(labels)  # [None, 17]
        input_length = tf.fill((BS,), self.pred2.shape[1])
        loss = tf.nn.ctc_loss(labels=tf.cast(self.targets, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
                              logits=self.pred2,
                              label_length=tf.cast(BS, tf.int32),
                              logit_length=input_length,  # tf.cast(tf.shape(pred2)[1],tf.int32),
                              logits_time_major=False,
                              blank_index=char2int_Attn_array.shape[0])  # (None,)
        loss = tf.reduce_mean(loss)
        self.loss_CTC = loss
        return loss
        #return pred2, output
########################## train #####################################
#from dataloader import  traingen,valgen#create_charset_ds ,
#df_train create_charset_ds()
#  [None, 17]
optimizer = tf.keras.optimizers.Adam( learning_rate=0.1)
GCTCInstance = GCTC()
#model_trans = GCTCInstance.forward_transformer()
x_sanity = tf.convert_to_tensor(tf.expand_dims(X_batch[0],axis=0)) #[1,64,411,3]
y_sanity = tf.convert_to_tensor(tf.expand_dims(y_batch_CTC[0], axis=0))
times_in = 1
#model_gctc.summary()
##########################################################


CTC = True
if CTC:
    #all_ds_in_out = [(x_sanity, y_sanity , times_in)]
    all_ds_in_out = [(X, ctc_label_extend, times_in)]
    GCTCInstance.forward_CTC()
    model_gctc = GCTCInstance.model_CTC_ok
    def train_data_for_one_epoch():
        losses =[]
        for step , (x_batch_train , y_batch_train, times) in enumerate(all_ds_in_out):
            logits , loss_val = apply_gradient(optimizer , model_gctc , x_batch_train, y_batch_train, times)
            losses.append(loss_val)
        return losses
    def CTCLoss(y_true, y_pred, times):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(times, dtype="int64")#tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        y_true1 = tf.where(tf.equal(y_true, -1) , 0, y_true)
        loss = tf.keras.backend.ctc_batch_cost(y_true1, y_pred, input_length, label_length)
        return loss

    def loss_function(real, pred, times_in):
        BS = tf.shape(real)[0]#self.labels_CTC)[0]
        # class targetbuild(tf.keras.layers):#Model):
        #     def __init__(self):
        #         super(targetbuild, self).__init__()
        #
        #     def call(self, labels):
        #         idx = tf.where(tf.not_equal(labels, -1))
        #         targets = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.cast(tf.shape(labels), tf.int64))
        #         return targets
        # targets = targetbuild()(real)#labels)  # [None, 17]
        labels= real
        idx = tf.where(tf.not_equal(labels, -1))
        targets = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.cast(tf.shape(labels), tf.int64))
       # targets =tf.convert_to_tensor(real)
       # targets1 = targets.values
        input_length = tf.cast(tf.fill((BS,), pred.shape[1]),tf.int32)
        times_in1 = tf.cast(tf.fill((BS,),times_in), tf.int32)
        #times_in1 = tf.cast(tf.fill((BS,),pred.shape[1]), tf.int32)
       # pred = tf.transpose(pred, [1, 0, 2])
        loss = tf.nn.ctc_loss(labels=tf.cast(targets, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
                              logits=pred,
                              label_length=None,#times_in,# tf.cast(BS, tf.int32),
                              logit_length=times_in1,  # tf.cast(tf.shape(pred2)[1],tf.int32),
                              logits_time_major=False,
                              blank_index=79)#char2int_Attn_array.shape[0])  # (None,)
        # pred = tf.transpose(pred , [1,0,2])
        # #loss = tf.compat.v1.nn.ctc_loss(targets, pred, times_in1, ignore_longer_outputs_than_inputs=True)
        # loss = tf.compat.v1.nn.ctc_loss(targets, pred, times_in1, ignore_longer_outputs_than_inputs=True)

        loss = tf.reduce_mean(loss)
        return loss
    optimizer = tf.keras.optimizers.Adam()
    def apply_gradient(optimizer, model , x, y, times_in):
        with tf.GradientTape() as tape:
            logits = model([x,y,times_in])

            loss_val = CTCLoss(y_true=y, y_pred=logits, times=times_in)
            # loss_val = loss_function(real =y, pred =logits, times_in = times_in)# times_in)
            variables = model.trainable_variables
        gradients = tape.gradient(loss_val ,variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return logits, loss_val
    #training loop
    epochs=1000
    for epoch in range(epochs):
        losses_train= train_data_for_one_epoch()

        #loss_val = validation()
        losses_train_mean = np.mean(losses_train)
        print(losses_train_mean)
        #loss_val_mean = np.mean(loss_val)
Attn = False
if Attn:
    #all_ds_in_out = [(x_sanity, y_sanity, times_in)]
    all_ds_in_out = [(X, ctc_label_extend, t)]
    GCTCInstance.forward_Attn()
    model_Attn = GCTCInstance.model_Attn

    def train_data_for_one_epoch():
        losses = []
        for step, (x_batch_train, y_batch_train, times) in enumerate(all_ds_in_out):
            logits, loss_val = apply_gradient(optimizer, model_gctc, x_batch_train, y_batch_train, times)
            losses.append(loss_val)
        return losses


    def loss_function(real, pred, times_in):
        BS = tf.shape(real)[0]  # self.labels_CTC)[0]

        class targetbuild(tf.keras.layers.Layer):
            def __init_(self):
                super(targetbuild, self).__init__()

            def call(self, labels):
                idx = tf.where(tf.not_equal(labels, -1))
                targets = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.cast(tf.shape(labels), tf.int64))
                return targets

        targets = targetbuild()(real)  # labels)  # [None, 17]
        input_length = tf.fill((BS,), pred.shape[1])
        loss = tf.nn.ctc_loss(labels=tf.cast(targets, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
                              logits=pred,
                              label_length=times_in,  # tf.cast(BS, tf.int32),
                              logit_length=input_length,  # tf.cast(tf.shape(pred2)[1],tf.int32),
                              logits_time_major=False,
                              blank_index=char2int_Attn_array.shape[0])  # (None,)
        loss = tf.reduce_mean(loss)
        return loss


    optimizer = tf.keras.optimizers.Adam()


    def apply_gradient(optimizer, model, x, y, times_in):
        with tf.GradientTape() as tape:
            logits = model([x, y, times_in])
            loss_val = loss_function(real=y, pred=logits, times_in=times_in)  # times_in)
            variables = model.trainable_variables
        gradients = tape.gradient(loss_val, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return logits, loss_val


    # training loop
    epochs = 500
    for epoch in range(epochs):
        losses_train = train_data_for_one_epoch()

        # loss_val = validation()
        losses_train_mean = np.mean(losses_train)
        print(losses_train_mean)
        # loss_val_mean = np.mean(loss_val)



loss_in = GCTCInstance.loss_CTC_calc()#(ytrue= , ypred=)
#model_gctc.compile(optimizer=optimizer, loss=loss_in)
model_gctc.compile(optimizer=optimizer, loss=loss_CTC_calc)
x_sanity = tf.convert_to_tensor(tf.expand_dims(X_batch[0],axis=0)) #[1,64,411,3]
y_sanity = tf.convert_to_tensor(tf.expand_dims(y_batch_CTC[0], axis=0))
times_in = 1
history = model_gctc.fit([x_sanity, y_sanity, times_in], y_sanity, batch_size=1, epochs=2)





##########################################################################
# _, model_Attn =GCTC().forward_Attn()
# model_Attn.compile(optimizer=optimizer, loss = tf.keras.losses.sparse_categorical_crossentropy(), metrics='acc' )#GCTC().loss)#tf.keras.losses.sparse_categorical_crossentropy() , metrics='acc')
# print(model_Attn.summary())
# model_Attn.fit([X_batch, y_batch_Attn],y_batch_Attn,  batch_size=1)
#GCTC().resnet_model.save_weights('')
#########################################
# #model_gctc = GCTCInstance.forward_CTC()
# GCTCInstance.model_CTC_ok.compile(optimizer=optimizer, loss=GCTCInstance.loss_CTC_calc)
# #model_gctc = GCTCInstance.model_CTC_ok
# #
# #loss = GCTCInstance.labels_CTC
# #model_gctc = GCTCInstance.model_CTC
#
# model_gctc.compile(optimizer=optimizer, loss=GCTCInstance.loss_CTC_calc)
# #self.model_CTC = model22
#
#
# model_trans = GCTCInstance.model_transformer
# model_gctc.compile(optimizer=optimizer)#, loss=lossCTC)  #already loss is applied as a layer
# model_gctc.fit([X_batch,y_batch_CTC] , y_batch_CTC, batch_size=2)
# model_trans = GCTCInstance.forward_transformer()
#
# #lossCTC = GCTCInstance.losses_CTC
# #model_trans = GCTCInstance.model_transformer
#
# model_gctc.compile(optimizer=optimizer)#, loss=lossCTC)  #already loss is applied as a layer
# model_gctc.fit([X_batch,y_batch_CTC] , y_batch_CTC, batch_size=2)
# print(model_gctc.summary())


#model1.save_weights('model.hdf5')

#model = tf.keras.models.load_model('myModel.h5')
#model.load_weights('my_model_weights.h5')



#model1.fit([X_batch,y_batch_CTC] , y_batch_CTC)
'''history = model1.fit(traingen,
                    epochs=100,
                    validation_data=valgen,
                    verbose=1,
                    #callbacks=callbacks_list,
                    shuffle=True)'''




'''history = model2.fit(traingen,
                    epochs=100,
                    validation_data=valgen,
                    verbose=1,
                    #callbacks=callbacks_list,
                    shuffle=True)'''
# model2 should be saved and used during testing
########################t test################################
'''model2 = tf.keras.models.load_model('path')


def preprocess(self, image):
    im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = im.shape[:2]
    resized = cv2.resize(im, (int(w * self.targetHeight / h), self.targetHeight))
    resized = np.asarray(resized, dtype=np.float)
    resized = resized / 255.0
    resized = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)
    return resized


img=cv2.imread('')
def preprocess(img):
    return img
img = preprocess(img)
model2.predict(img)
####################################################################
# try for GCN

def norm_adjacency_matrix(A):
    I = tf.eye(tf.shape(A)[0])
    A_hat = A + I
    D_inv = tf.linalg.tensor_diag(
        tf.pow(tf.reduce_sum(A_hat, 0), tf.cast(-0.5, tf.float32)))
    D_inv = tf.where(tf.math.is_inf(D_inv), tf.zeros_like(D_inv), D_inv)
    A_hat = D_inv @ A_hat @ D_inv
    return A_hat
def distance_matrix(H):  # H is feature vector (Bs , w, c*h))
    dim_d = H.shape[-2]  # or 0 [1]
    A_D = D = np.zeros((dim_d, dim_d))
    for i in range(dim_d):  # len(D.shape[0])):
        for j in range(dim_d):  # len(D.shape[1])):
            beta = 0.3  # scale factor  # it should be hyperparameter
            D[i, j] = np.abs(i - j)
            A_D[i, j] = (np.exp(-D[i, j] + beta)) / (np.exp(-D[i, j] + beta) + 1)
    return A_D
def similarity_matrix(D):
    f = lambda x, y: ((np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)))
    def linear_transform(a):  # c is linear transformation result of h
        return a
        # pass

    # D=np.zeros((4,5))
    shD = D.shape[-2]  # 1
    A_S = np.zeros((shD, shD))
    for i in range(shD):
        for j in range(shD):
            A_S[i, j] = list(map(f, (linear_transform(D[i]), linear_transform(D[j]))))

    L2Norm = tf.sqrt(tf.reduce_sum(tf.pow(h, 2), axis=-1))
    h = h / tf.tile(tf.expand_dims(L2Norm, axis=-1), [1, 1, sh[-1]])
    A_S = tf.matmul(h, tf.transpose(h, [0, 2, 1]))
    return A_S


# adjacency_matrix = lambda h: similarity_matrix(h)*distance_matrix(h)
# return adjacency_matrix
from keras import backend as K

# with a Sequential model
# layer_output = K.function([model.layers[0].input],
#                                  [model.layers[3].output])
# layer_output_h = layer_output([x])[0]
l1 = tf.keras.layers.Lambda(lambda h: similarity_matrix(h) * distance_matrix(h) * h)(h)
# TODO : W_g
w = 3
dense1 = tf.keras.layers.Dense(w)(l1)  # or i can add it as the first stage in BiLTSM
# model_GCN=Sequential([l1, dense1])'''

# encoder input : CNN output
# decoder input : target label
# decoder weight : a mask

CNNoutput =1
decoder_inputs=2
target_weights=3


'''def Seq2SeqDynamicModel(encoder_inputs_tensor, decoder_inputs, target_weights, target_vocab_size, buckets,
                        target_embedding_size, attn_num_layers):
    pass


attention_decoder_model = Seq2SeqDynamicModel(
            encoder_inputs_tensor=CNNoutput,
            decoder_inputs=decoder_inputs,
            target_weights=target_weights,
            target_vocab_size=10, #config.num_classes,
            buckets=3,#self.buckets,
            target_embedding_size=128,#target_embedding_size,
            attn_num_layers=2,#attn_num_layers,
            attn_num_hidden=512)#attn_num_hidden,
            #forward_only=not(is_training))

output_len = len(attention_decoder_model.output)
num_feed=[]
prb_feed=[]
for line in range(output_len):
    guess = tf.argmax(attention_decoder_model.output[line], axis=1)
    proba = tf.reduce_max(tf.nn.softmax(attention_decoder_model.output[line]), axis=1)
    num_feed.append(guess)
    prb_feed.append(proba)

################################
def linear(input, prev_attn):
    tf.python.ops
    from tensorflow.contrib.rnn.python.ops import rnn_cell_impl
    from tensorflow.contrib.rnn.python.ops import rnn_cell_impl
    pass


def cell(input, prev_attn, prev_state):
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(
        num_units,
        activation=None,
        reuse=None,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        dtype=None,
        **kwargs)
    return cell
(cell_output, new_state) = cell(linear(inputs, pre_att_mask), prev_state)
#    A = linear(input , prev_attn)
#    return cell_outputr, new_state
new_attn = tf.math.softmax(V.T @ tf.tanh(u @ new_state + w @ attn_state))
output = linear(new_attn , cell_output)


def attention(new_state):  #new_state =q
    new_attn = tf.math.softmax(V.T @ tf.tanh(u @ new_state + w @ attn_state))
    context_vector =new_attn

    return context_vector , attention_weight


attn , attn_weight = attention(new_state)'''

####################
def loss_CTC_calc(ytrue, ypred): #len(char2int_Attn_array)  #times should also be considered
#    ypred = self.pred2
    labels = ytrue
    BS =labels.get_shape()[0]# tf.shape(labels)[0]
    class targetbuild(tf.keras.layers.Layer):#layers.Layer):models.Model
        def __init_(self):
            super(targetbuild, self).__init__()

        def call(self, labels):
            idx = tf.where(tf.not_equal(labels, -1))
            targets = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.cast(tf.shape(labels), tf.int64))
            return targets

    targets = tf.cast(targetbuild()(labels), tf.int32) #targetbuild.predict()
    input_length = tf.cast(tf.fill((BS,), ypred.get_shape()[1]), tf.int32)
    # loss = tf.nn.ctc_loss(labels=tf.cast(targets, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
    #                       logits=tf.transpose(ypred,[1,0,2]),
    #                       label_length=tf.cast(BS, tf.int32),  # i should change it by times
    #                       logit_length=input_length,  # tf.cast(tf.shape(pred2)[1],tf.int32),
    #                       logits_time_major=False,
    #                       blank_index=len(char2int_Attn_array))  # (None,)
    times = tf.cast(tf.tile(tf.expand_dims(seq_len, 0), [BS]), dtype=tf.int32)
    loss = tf.compat.v1.nn.ctc_loss(targets,
                          ypred,
                          times,  # should be times
                          preprocess_collapse_repeated=False,
                          ctc_merge_repeated=True,
                          ignore_longer_outputs_than_inputs=False,
                          time_major=True)


    '''ctc_loss = tf.nn.ctc_loss(
        gt,  # labels
        model_output,  # logits
        seq_len,  # label_length
        input_length,  # logits_length
        logits_time_major=False,
        blank_index=len(vocab))#self.vocab)    )'''
    loss = tf.reduce_mean(loss)
    print(loss)
    return loss