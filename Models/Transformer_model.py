import tensorflow as tf
from ConFig.Config import ConfigReader
cfg = ConfigReader()
#from Models.modules.Layers import Transformer  # some modification to Gitlab
from tryTransformer import Transformer  # this is Original GitLab one


#from Models.modules.ResNet import ResNet_backbone, ResNet50, NewResNet,ResNetLayer,mobilenet  # i change it since the previous one did not give the trainable variables

#from DataLoad.DataLoadFunc import vocab_CTC, Maxlen_MJ, maxW
from DataLoad.TrainDataLoading import GetData
DS , vocab, Maxlen ,maxW  = GetData(dataset='Syntext', ctc=False, N_sample=1, batchsize=1)
from Models.modules.callbacks import CustomSchedule
from res_and_ctc import NewResNet

#maxW=650
class OverallModel_Transformer():
    def __init__(self,maxW, lenvocab, maxlen, warmup):
        super(OverallModel_Transformer, self).__init__()
        #self.embed = ResNet_backbone()
        self.warmup=warmup

        self.num_layers = cfg.num_heads_tr
        self.d_model = cfg.d_model_tr
        self.num_heads = cfg.num_heads_tr
        self.dff = cfg.dff_tr
        self.dropout_rate = cfg.dropout_rate_tr

        self.FE = NewResNet()#ResNet_backbone #Resnet50()
        self.FE()
        self.FEnew = self.FE.model
        #self.FElayer =ResNetLayer
        #self.FEnew =mobilenet()
        self.vocab =vocab
        self.maxW = maxW
        self.maxlen = maxlen
      #  BS = tf.shape(input)[0]
        #chars_sorted = {'a', 'b', 'r'}
        self.trans = Transformer(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads,  dff=self.dff,
                                 target_vocab_size=lenvocab+1,#len(chars_sorted) + 1,     # self.dataLoader.vocab) + 1, # for PAD add " + 1"
                                 maxSourceLength=tf.divide(tf.cast(self.maxW, tf.float32), cfg.SeqDivider),#150,#input.get_shape()[2],  # self.dataLoader.maxSeqLen, 12
                                 maxTargetLength=self.maxlen-1,#labels.get_shape()[1] - 1, 27
                                 rate=self.dropout_rate)  # maxTargetLength=self.dataLoader.targetMaxLen - 1,
        self.customlr = CustomSchedule(d_model=self.d_model)
        if self.warmup:
            self.learning_rate = self.customlr
        else:
            self.learning_rate = cfg.lr_tr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)
        self.debug=False
    def __call__(self, x):#,y):
        end_points = {}
        x1,y1,encoding_mask =x

        #x1 = tf.squeeze(x1, axis=1)
        if self.debug:
            end_points["sq1"] = x1
#        encoding_mask=tf.squeeze(encoding_mask, axis =1)
        # three kind of embedding : mobilenet/resnet here/resnet from GCTC
        #x_in = model1(x1)
        # if self.debug:
        #     end_points["m1"] = x_in
        x_in = self.FEnew(x1)
       # modelResNet = tf.keras.Model(inputs=x1 ,outputs =x_in)
#       # x4 = gc.feature_extraction(x1)
        pred , att = self.trans([x_in,y1,encoding_mask],training=True)

        return pred #, resmodel #end_points


class FULL_FE_TRANS():
    def __init__(self, maxW, lenvocab, maxlen, warmup=False):
        super(FULL_FE_TRANS, self).__init__()
        # self.embed = ResNet_backbone()
        self.num_layers = cfg.num_heads_tr
        self.d_model = cfg.d_model_tr
        self.num_heads = cfg.num_heads_tr
        self.dff = cfg.dff_tr
        self.dropout_rate = cfg.dropout_rate_tr
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.FE = NewResNet() #ResNet50()#NewResNet()  # ResNet_backbone #Resnet50()
        #self.FEnew = self.FE().model
        #self.FElayer = ResNetLayer
        # self.FEnew =mobilenet()
        self.vocab = vocab
        self.maxW = maxW
        self.maxlen = maxlen
        self.x1 = tf.keras.layers.Input(shape=(cfg.targetHeight,None,1))
        self.y1 = tf.keras.layers.Input(shape =[None])
        self.encoding_mask = tf.keras.layers.Input([None])
        #  BS = tf.shape(input)[0]
        # chars_sorted = {'a', 'b', 'r'}
        self.trans = Transformer(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads,
                                 dff=self.dff,
                                 target_vocab_size=lenvocab + 1,
                                 # len(chars_sorted) + 1,     # self.dataLoader.vocab) + 1, # for PAD add " + 1"
                                 maxSourceLength=tf.divide(tf.cast(self.maxW, tf.float32), cfg.SeqDivider),
                                 # 150,#input.get_shape()[2],  # self.dataLoader.maxSeqLen, 12
                                 maxTargetLength=self.maxlen - 1,  # labels.get_shape()[1] - 1, 27
                                 rate=self.dropout_rate)  # maxTargetLength=self.dataLoader.targetMaxLen - 1,
        self.debug = False
        self.model = None
        self.warmup = warmup
        self.customlr = CustomSchedule(d_model=self.d_model)
        if self.warmup:
            self.learning_rate = self.customlr
        else :
            self.learning_rate = cfg.lr_tr

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def __call__(self):  # ,y):
        end_points = {}
        #x1, y1, encoding_mask = x

        # x1 = tf.squeeze(x1, axis=1)
        if self.debug:
            pass#@ end_points["sq1"] = x1
        #        encoding_mask=tf.squeeze(encoding_mask, axis =1)
        # three kind of embedding : mobilenet/resnet here/resnet from GCTC
        # x_in = model1(x1)
        # if self.debug: x_in = self.FEnew(self.x1)
        #     end_points["m1"] = x_in
       # x_in = self.FEnew(self.x1)
        x_in = self.FE(self.x1)


        # modelResNet = tf.keras.Model(inputs=x1 ,outputs =x_in)
        #       # x4 = gc.feature_extraction(x1)
        pred, att = self.trans([x_in, self.y1, self.encoding_mask], training=True)
        model1 = tf.keras.Model(inputs=[self.x1,self.y1,self.encoding_mask], outputs=pred)  # output)
        self.model = model1  # tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)#output)
        return self  # , softmax_output

