import tensorflow as tf
from Models.modules.ResNet import ResNet_backbone, ResNet50, NewResNet
from Models.modules.Layers import GraphConvolutionLayer, BiLSTM_func , BiLSTM_func1
#from DataLoad.DataLoadFunc import vocab_CTC, Maxlen_MJ
from ConFig.Config import ConfigReader
cfg = ConfigReader()
from Models.modules.Layers import  ADJlayer
def ctc_lambda_func(args):
    iy_pred, ilabels, iinput_length, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return tf.keras.backend.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)

class CTC_model(tf.keras.Model):
    def __init__(self):
        super(CTC_model, self).__init__()
        self.vocab_size =77
        self.feature_extraction = ResNet_backbone
        self.GCN =GraphConvolutionLayer(units =500)
        self.BiLSTM = BiLSTM_func
        self.classifier = tf.keras.layers.Dense(self.vocab_size+1, activation ='softmax')
        #  self.input_image = tf.keras.layers.Input(shape=(64,600,3))
        #  self.labels_CTC = tf.keras.layers.Input(shape=(25))
        #  self.times = tf.keras.layers.Input(shape=())
        self.decoder = tf.nn.ctc_greedy_decoder
        self.ctc_prediction_len =30



    def __call__(self, x1):
        x , y, times = x1
        feat_vec = self.feature_extraction(x) #([1, 151, 2048])
        #GCNout = self.GCN(feat_vec)
       # RNNout = self.BiLSTM(GCNout, times)
        logits = self.classifier(feat_vec)  #([1, 151, 78])
        pred_trans = tf.transpose(logits, [1, 0, 2])
        Sh1 = tf.shape(y)[0]
        seq_lengths = tf.fill([Sh1], tf.shape(pred_trans)[0])
        seq_lengths = tf.fill([Sh1], times)
        times = tf.fill([Sh1], times)
        decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(pred_trans,
                                                                    tf.cast(times, tf.int32),
                                                                    merge_repeated=False,
                                                                    top_paths=30)
        decoded =tf.nn.ctc_greedy_decoder(pred_trans, sequence_length=tf.cast(seq_lengths , tf.int32), blank_index=-1)# )
        res_lab = tf.compat.v1.sparse_to_dense(tf.cast(decoded[0][0].indices, tf.int32),  # none,90
                                               tf.stack([tf.shape(y)[0],
                                                         self.ctc_prediction_len]), decoded[0][0].values,
                                               default_value=-1)
        D = tf.keras.backend.ctc_decode(
            logits,
            tf.fill([Sh1], 79),
            greedy=True,
            beam_width=100,
            top_paths=1
        )
        decode = False
        if decode :

            from test_CTC import vocab
            import numpy as np
            chars_sorted = sorted(vocab)
            chars_sorted = list(chars_sorted)
            char_array_CTC = np.array(chars_sorted)
            A = res_lab.numpy()
            [char_array_CTC[i] for i in A]


        tf.as_string(res_lab[0])

        return logits


from Models.modules.ResNet import  ResNet_DL , ResNet_backbone
from Models.modules.Layers import CTCLayer_DL
#
class CTC_model_DL(tf.keras.Model):
    def __init__(self , after_transformer_train = False, vocab_size =93, maxlen =61, input_shape=0 , label_shape=0 ):
        self.vocab_size = vocab_size# 57#77
        super(CTC_model_DL, self).__init__()
        self.input1 = tf.keras.layers.Input(shape=input_shape, name="image") #(32,128,1),can i change to None?(cfg.targetHeight,None,3)
        self.label = tf.keras.layers.Input(name="label", shape=(label_shape,), dtype="float32")#(None,)
        self.Feat = ResNet50()#ResNet_backbone
        self.GCN =GraphConvolutionLayer(units =2048)
        self.BiLSTM = BiLSTM_func
        self.classifier = tf.keras.layers.Dense(self.vocab_size+1, activation ='softmax')
        self.model = None
        self.after_transformer_train = after_transformer_train
        self.optimizer = tf.keras.optimizers.Adam(lr=cfg.lr, beta_1=0.99, beta_2=0.999, clipnorm=1.0)
        self.maxlen=maxlen
    def __call__(self):
        if self.after_transformer_train:
            self.Feat.trainable = False
            self.Feat.Conv1.trainable =False
            self.Feat.idenblock.trainable= False
            self.Feat.resblock.trainable =False

            #
            self.Feat.load_weights('')

        feat  = self.Feat(self.input1)  #[none 30 2048], resnet_model

        GCNin = feat#resnet_model.output
        GCNout = self.GCN(feat , GCNin)
        RNNout = self.BiLSTM(GCNout,self.maxlen) #19
        softmax_output =self.classifier(RNNout)#RNNout) #tf.keras.layers.Dense(78, activation='softmax')(softmax_output)
        self.softmax_output = softmax_output
        output = CTCLayer_DL(name="ctc_loss")(self.label, softmax_output)
        model1 = tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)  # output)
        self.model = model1#tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)#output)
        return self#, softmax_output


class CTC_model_DL_new(tf.keras.Model):
    def __init__(self):
    #    self.vocab_size = len(vocab_CTC) #77
        super(CTC_model_DL_new, self).__init__()
        #self.input1 = tf.keras.layers.Input(shape=(32,128,1), name="image")
       # self.label = tf.keras.layers.Input(name="label", shape=(None,), dtype="float32")
        self.Feat = ResNet50()#ResNet_backbone#ResNet50#
        self.GCN  = GraphConvolutionLayer(units =2048)
        self.BiLSTM = BiLSTM_func
        self.classifier = tf.keras.layers.Dense(self.vocab_size+1, activation ='softmax')
        self.model = None
    def __call__(self,x):
        x1,y1 =x
      #  x =tf.squeeze(x1, axis=1)
        feat  = self.Feat(x1)#self.input1)  #[none 30 2048], resnet_model
        GCNin = feat#resnet_model.output
        GCNout = self.GCN(feat , GCNin)
        RNNout = self.BiLSTM(GCNout,19)#Maxlen_MJ)
        softmax_output =self.classifier(GCNin)#RNNout) #tf.keras.layers.Dense(78, activation='softmax')(softmax_output)
        self.softmax_output = softmax_output
       # output = CTCLayer_DL(name="ctc_loss")(self.label, softmax_output)
      #  model1 = tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)  # output)
       # self.model = model1#tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)#output)
        return softmax_output#

#
# def buildCTC():#_model_DL(tf.keras.Model):
#     input1 = tf.keras.layers.Input(shape=(cfg.targetHeight,400,1), name="image") #(32,128,1),can i change to None?
#     label = tf.keras.layers.Input(name="label", shape=(None,), dtype="float32")
#     Feat = ResNet50()#ResNet_backbone
#     GCN =GraphConvolutionLayer(units =2048)
#     BiLSTM = BiLSTM_func
#     classifier = tf.keras.layers.Dense(93, activation ='softmax')
#     #elf.model = None
#      #   self.after_transformer_train = after_transformer_train
#     optimizer = tf.keras.optimizers.Adam(lr=cfg.lr, beta_1=0.99, beta_2=0.999, clipnorm=1.0)
#     #    self.maxlen=maxlen
#     #
#     #.Feat.load_weights('')
#     feat  = Feat(input1)  #[none 30 2048], resnet_model
#
#     GCNin = feat#resnet_model.output
#     GCNout = GCN(feat , GCNin)
#     RNNout = BiLSTM(GCNout,61) #19
#     softmax_output =classifier(RNNout)#RNNout) #tf.keras.layers.Dense(78, activation='softmax')(softmax_output)
#     softmax_output = softmax_output
#     output = CTCLayer_DL(name="ctc_loss")(label, softmax_output)
#     model1 = tf.keras.Model(inputs=[input1, label], outputs=softmax_output)  # output)
#     model = model1#tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)#output)
#     return model#self#, softmax_output
class modelCTC(tf.keras.Model):
    def __init__(self,lenvoc =3):
        super(modelCTC,self).__init__()
        self.lenvoc =lenvoc
        self.classifier = tf.keras.layers.Dense(self.lenvoc+ 1, activation='softmax')
        self.inputShape = tf.keras.layers.Input((cfg.targetHeight, None, 1),
                                           dtype='float32')  # (None, 64, 3))  # base on Tensorflow backend
        self.labels = tf.keras.layers.Input(name='the_labels', shape=[None],
                                       dtype='float32')  # , ragged=True )# [None] label_shape

        self.input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')  # , ragged=True )
        self.label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')  # , ragged=True )
        self.AS = tf.keras.layers.Dot(axes=2, normalize=True)
        self.AD = ADJlayer()
        self.Feat = ResNet50()
        self.BILSTM = BiLSTM_func
        self.model =None
        self.optimizer = tf.keras.optimizers.Adam(lr=0.00015, beta_1=0.99, beta_2=0.999, clipnorm=5.0)

    def __call__(self):
        X = self.Feat(self.inputShape )
        #print(X.shape)
        A_D = tf.cast(self.AD(X), tf.float32)
        #print('AD',A_D.shape)
        A_S = self.AS([X,X])
        A = A_S @ A_D
        X = A @ X  # @ self.W
        X = self.BILSTM(X, self.input_length)
        X = self.classifier(X)
        loss_out = tf.keras.layers.Lambda(ctc_lambda_func,
                                          output_shape=(1,),
                                          name='ctc')([X, self.labels, self.input_length, self.label_length])
        model1 = tf.keras.Model(inputs=[self.inputShape, self.labels, self.input_length, self.label_length],
                               outputs=[X])
        self.model =model1
        return self

class CTCalone(tf.keras.Model):
    def __init__(self, lenvoc):
        super(CTCalone, self).__init__()

        self.lenvoc = lenvoc
        self.AD = ADJlayer()
        self.BILSTM = BiLSTM_func
        self.AS = tf.keras.layers.Dot(axes=2, normalize=True)
        self.classifier = tf.keras.layers.Dense(self.lenvoc + 1, activation='softmax')

    def __call__(self, X, label , inputlength, labellength):
        A_D = tf.cast(self.AD(X), tf.float32)
        # print('AD',A_D.shape)
        A_S = self.AS([X, X])
        A = A_S @ A_D
        X = A @ X  # @ self.W
        inputlength1 =tf.expand_dims(inputlength, axis=-1)
        X = self.BILSTM(X, inputlength1)
        XX = self.classifier(X)
        labellength1 =tf.expand_dims(labellength, axis=-1)
        loss_out = tf.keras.layers.Lambda(ctc_lambda_func,
                                          output_shape=(1,),
                                          name='ctc')([X, label , inputlength1, labellength1])#self.labels, self.input_length, self.label_length])

        return XX


class modelCTC1(tf.keras.Model):
    def __init__(self,lenvoc =4):
        super(modelCTC1,self).__init__()
        self.lenvoc =lenvoc
        self.classifier = tf.keras.layers.Dense(self.lenvoc+ 1, activation='softmax')
        self.inputShape = tf.keras.layers.Input((None, 2048),
                                           dtype='float32')  # (None, 64, 3))  # base on Tensorflow backend
        self.labels = tf.keras.layers.Input(name='the_labels', shape=[None],
                                       dtype='float32')  # , ragged=True )# [None] label_shape

        self.input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')  # , ragged=True )
        self.label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')  # , ragged=True )
        self.AS = tf.keras.layers.Dot(axes=2, normalize=True)
        self.AD = ADJlayer()
        self.Feat = ResNet50()
        self.BILSTM = BiLSTM_func1
        self.model =None
        self.optimizer = tf.keras.optimizers.Adam(lr=0.00015, beta_1=0.99, beta_2=0.999, clipnorm=5.0)

    def __call__(self):
      #  X = self.Feat(self.inputShape )
        #print(X.shape)
        X = self.inputShape
        label =self.labels
        inputlength = self.input_length
        labellength = self.label_length

        A_D = tf.cast(self.AD(X), tf.float32)
        # print('AD',A_D.shape)
        A_S = self.AS([X, X])
        A = A_S @ A_D
       # A= tf.squeeze(A, 2)
        X = A @ X  # @ self.W

       #
        X = self.BILSTM(X, inputlength)
        XX = self.classifier(X)
        labellength1 = tf.expand_dims(labellength, axis=-1)

        inputlength1 = tf.expand_dims(inputlength, axis=-1)
        loss_out = tf.keras.layers.Lambda(ctc_lambda_func,
                                      output_shape=(1,),
                                      name='ctc')([XX, label, self.input_length, self.label_length])#inputlength1, labellength1])

        self.model = tf.keras.Model(inputs=[self.inputShape, self.labels, self.input_length, self.label_length],
                               outputs=[XX])

        return self
NewResNet1 = NewResNet()
#NewResNet1

class OverallCTC():
    def __init__(self, lenvoc,aftertraintransformer=False):
        super(OverallCTC, self).__init__()
       # self.FE = ResNet50()  # NewResNet()  # ResNet_backbone #Resnet50()
        self.FEnew = NewResNet1.model#NewResNet().model  # ResNet_backbone #Resnet50()
        #self.FEnew = self.FE().model
        self.ctcmodel = modelCTC1(lenvoc =lenvoc).model#CTCalone(lenvoc)
        #self.CTC = self.ctcmodel11().model
        self.aftertraintransformer =  aftertraintransformer
        self.lenvoc = lenvoc


    def __call__(self, x):
        X, label, inputlength, labellength = x
        if self.aftertraintransformer :
            self.FEnew.load_model("weights.hdf5")
            for layer in self.FEnew.layers:
                layer.trainable = False

        out = self.FEnew(X)  #[1, 34,2048]
        with_ctc = False
        if with_ctc:
            ctcout =self.ctcmodel(out, label , inputlength, labellength)
            return ctcout
        else:
          #  seqlen = tf.shape(x)[1]
            #x = tf.keras.layers.Dense(seqlen, activation="softmax")(out)

            #x1 = tf.keras.layers.Dense(10)(out)
            layer = tf.keras.layers.Dense(self.lenvoc+1, activation='softmax')
            x1 = tf.keras.layers.TimeDistributed(layer)(out)#tf.keras.layers.Dense(8, activation='softmax'))(out)
            #x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmax(x, axis=-1))(x)
            return x1
            #ctcout =self.CTC([out, label , inputlength, labellength])

        #return cnnout#ctcout
loss = tf.keras.losses.CategoricalCrossentropy()#tf.keras.losses.categorical_crossentropy()
#ctc  = OverallCTC(70)
import numpy as np
# X = np.load('X_batch.npy')
# y = np.load('y_batch_Attn.npy')
# ctc(X,y)
#OverallCTC()



