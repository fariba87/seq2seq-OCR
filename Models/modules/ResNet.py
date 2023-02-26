import tensorflow as tf
def residual_block1(res_n_ch, x):
    a, b, c = res_n_ch
    '''model_res_block = tf.keras.models.Sequential([tf.keras.layers.Conv2D(a, (1, 1)),
                                                  tf.keras.layers.Conv2D(b, (3, 3)),
                                                  tf.keras.layers.Conv2D(c, (1, 1))])'''
    shortcut = x
    X1 = tf.keras.layers.Conv2D(a, (1, 1),strides =(4,1), padding='same')(x)# valid # i add it myself
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(b, (3, 3), strides =(2,1), padding='same')(X1)  # i add it stride
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1) ####

    X_shortcut = tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(shortcut)#valid
    X_shortcut = tf.keras.layers.Activation('relu')(X_shortcut)#####
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X1 = tf.keras.layers.Add()([X_shortcut, X1])
    Xo = tf.keras.layers.Activation('relu')(X1)
    return Xo

def identity_block(res_n_ch, X):
    #  conv_name_base = 'res' + str(stage) + block + '_branch'
    #  bn_name_base = 'bn' + str(stage) + block + '_branch'
    a, b, c = res_n_ch
    shortcut = X
    X1 = tf.keras.layers.Conv2D(a, (1, 1), padding='same')(X)#
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(b, (3, 3),  padding='same')(X1)
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)#
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X = tf.keras.layers.Add()([shortcut, X1])  # ?
    X = tf.keras.layers.Activation('relu')(X)
    return X

class ResNet50(tf.keras.Model):#tf.keras.layers.Layer): #
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resblock =residual_block1
        self.idenblock = identity_block
        self.Conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(10, 8), padding='same') # i add it# raw2-?8


    def __call__(self,x):
        res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
        # Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
        X = self.Conv1(x )#tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)  #i add the stride
        X = tf.keras.layers.BatchNormalization(axis=3)(X)  #
        X = tf.keras.layers.ReLU()(X)  #
        X = tf.keras.layers.MaxPool2D((6, 3), (2, 4), padding='same')(X)  #3 raw 2-?4

        X = self.resblock(res_n_ch[0], X)
        X = self.idenblock(res_n_ch[0], X)
        X = self.idenblock(res_n_ch[0], X)

        X = self.resblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = tf.keras.layers.MaxPool2D((8, 1), (8, 1),padding='same')(X)#2

        X = self.resblock(res_n_ch[2], X)
        X = self.idenblock(res_n_ch[2], X)
        X = self.idenblock(res_n_ch[2], X)
        X = tf.keras.layers.MaxPool2D((6, 1), (6, 1),padding='same')(X)#2

        X = residual_block1(res_n_ch[3], X)
        X = self.idenblock(res_n_ch[3], X)
        X = self.idenblock(res_n_ch[3], X)  # 3, 38 ,256
        # X = tf.keras.layers.ZeroPadding2D((1, 1))(X)
        # Xo = tf.keras.layers.AvgPool2D((4, 1), (1, 1))(X)  # 2,40,256
        # Xo = tf.keras.layers.AvgPool2D((2, 1), (1, 1))(Xo)  # 2,40,256
        Xo = tf.reshape(tf.transpose(X, [0, 2, 1, 3]), [-1, tf.shape(X)[2], tf.shape(X)[1]*tf.shape(X)[3]])#X.shape[2], X.shape[1] * X.shape[3]])
        #model = tf.keras.Model(inputs= x , outputs = Xo)
        return Xo #, model

def ResNet_backbone( Xin):
    res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
    # Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(Xin)  #
    X = tf.keras.layers.BatchNormalization(axis=3)(X)  #
    X = tf.keras.layers.ReLU()(X)  #
    X = tf.keras.layers.MaxPool2D((3, 3), (2, 2))(X)  #

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
    # X = tf.keras.layers.ZeroPadding2D((1, 1))(X)
    # Xo = tf.keras.layers.AvgPool2D((4, 1), (1, 1))(X)  # 2,40,256
    # Xo = tf.keras.layers.AvgPool2D((2, 1), (1, 1))(Xo)  # 2,40,256
    Xo = tf.reshape(tf.transpose(X, [0, 2, 1, 3]), [-1, X.shape[2], X.shape[1] * X.shape[3]])
    # Xo = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 1))(
    #     Xo)  # 40 ,256 # or use feature_map_to_featurevec function
    resnet_model = tf.keras.models.Model(inputs=Xin, outputs=Xo)
    #resnet_model
    return resnet_model # Xo #,


def ResNet_DL(inputs):
    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation="selu", padding='same')(inputs)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding='same')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)

    conv_3 = tf.keras.layers.Conv2D(128, (3, 3), activation="selu", padding='same')(pool_2)
    conv_4 = tf.keras.layers.Conv2D(128, (3, 3), activation="selu", padding='same')(conv_3)

    pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = tf.keras.layers.Conv2D(256, (3, 3), activation="selu", padding='same')(pool_4)

    # Batch normalization layer
    batch_norm_5 = tf.keras.layers.BatchNormalization()(conv_5)

    conv_6 = tf.keras.layers.Conv2D(256, (3, 3), activation="selu", padding='same')(batch_norm_5)
    batch_norm_6 = tf.keras.layers.BatchNormalization()(conv_6)
    pool_6 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = tf.keras.layers.Conv2D(64, (2, 2), activation="selu", padding ='same')(pool_6)

    squeezed = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(conv_7) # [none 31,64]

    # bidirectional LSTM layers with units=128
    blstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(squeezed)
    blstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(blstm_1)
    softmax_output = tf.keras.layers.Dense(78, activation='softmax', name="dense")(blstm_2)
    return softmax_output

class dw_sep_block(tf.keras.layers.Layer):
    def __init__(self, channels, alpha, strides, is_training, name=''):
        """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
        super(dw_sep_block, self).__init__()
        self.channels = int(channels * alpha)
        self.dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                            strides=strides,
                            use_bias=False,
                            padding='same',
                            name='{}_dw'.format(name))#,
                            #depthwise_initializer=initializer
                                         #)
        self.bn1 = tf.keras.layers.BatchNormalization(name='{}_bn1'.format(name), trainable=is_training)
        self.act1 = tf.keras.layers.ReLU(name='{}_act1'.format(name))

        # Pointwise
        self.pw = tf.keras.layers.Conv2D(self.channels,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   use_bias=False,
                   padding='same',
                   name='{}_pw'.format(name))#,
                   #kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization(name='{}_bn2'.format(name), trainable=is_training)
        self.act2 = tf.keras.layers.ReLU(name='{}_act2'.format(name))


    def __call__(self, x):
        # Depthwise
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        # Pointwise
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class mobilenet(tf.keras.layers.Layer):
    def __init__(self, is_training=True, alpha=1.0):
        super(mobilenet, self).__init__()
        self.alpha = alpha
        self.layersArc = [
            (64, (1, 1)),
            (128, (2, 2)),
            (128, (1, 1)),
            (256, (2, 2)),
            (256, (1, 1)),
            (512, (2, 2)),
            *[(512, (1, 1)) for _ in range(1)],
            (1024, (1, 1)),
            (1024, (1, 1))
        ]
        self.blocks = [dw_sep_block(channels, alpha, strides, is_training=is_training, name='block{}'.format(i)) for i, (channels, strides) in enumerate(self.layersArc)]
        self.avgPool = tf.keras.layers.AveragePooling2D(pool_size=[2, 1], strides=[2, 1])
        self.initial_conv = tf.keras.layers.Conv2D(int(32 * self.alpha),
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          use_bias=False,
                                          padding='same',
                                          name='initial_conv')#,
                                         # kernel_initializer=initializer)
        self.initial_bn = tf.keras.layers.BatchNormalization(name='initial_bn', trainable=is_training)
        self.initial_act = tf.keras.layers.ReLU(name='initial_act')

    def __call__(self, x_in):
        x = self.initial_conv(x_in)
        x = self.initial_bn(x)
        x = self.initial_act(x)

        for i in range(len(self.layersArc)):
            x = self.blocks[i](x)
        # x = self.avgPool(x)

        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, [0, 2, 1, 3])
        seq_len = tf.shape(x)[1]
        embeddings = tf.reshape(x, [batch_size, seq_len, -1])
        return embeddings

from ConFig.Config import ConfigReader
cfg = ConfigReader()
class NewResNet(tf.keras.Model):
    def __init__(self , after_transformer_train = False, input_shape=1):
        #self.vocab_size = vocab_size# 57#77
        super(NewResNet, self).__init__()
        self.input1 = tf.keras.layers.Input(shape=(cfg.targetHeight,None,1), name="image") #(32,128,1),can i change to None?

        self.Feat = ResNet50()#ResNet_backbone
        self.optimizer = tf.keras.optimizers.Adam(lr=cfg.lr, beta_1=0.99, beta_2=0.999, clipnorm=1.0)
        self.model =None

    def __call__(self):


        feat  = self.Feat(self.input1)  #[none 30 2048], resnet_model

        model1 = tf.keras.Model(inputs=self.input1, outputs=feat)  # output)
        self.model = model1#tf.keras.Model(inputs=[self.input1, self.label], outputs=softmax_output)#output)
        return self#, softmax_output


#################################################
class ResNetLayer(tf.keras.layers.Layer):
    def __init__(self , after_transformer_train = False, input_shape=1):
        #self.vocab_size = vocab_size# 57#77
        super(ResNetLayer, self).__init__()
        #self.input1 = tf.keras.layers.Input(shape=(cfg.targetHeight,None,1), name="image") #(32,128,1),can i change to None?
        self.resblock =residual_block1
        self.idenblock = identity_block
        self.Conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        #self.optimizer = tf.keras.optimizers.Adam(lr=cfg.lr, beta_1=0.99, beta_2=0.999, clipnorm=1.0)

    def __call__(self,x):
        res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
        # Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
        X = self.Conv1(x)#tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)  #
        X = tf.keras.layers.BatchNormalization(axis=3)(X)  #
        X = tf.keras.layers.ReLU()(X)  #
        X = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='same')(X)  #

        X = self.resblock(res_n_ch[0], X)
        X = self.idenblock(res_n_ch[0], X)
        X = self.idenblock(res_n_ch[0], X)

        X = self.resblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = self.idenblock(res_n_ch[1], X)
        X = tf.keras.layers.MaxPool2D((2, 1), (2, 1),padding='same')(X)

        X = self.resblock(res_n_ch[2], X)
        X = self.idenblock(res_n_ch[2], X)
        X = self.idenblock(res_n_ch[2], X)
        X = tf.keras.layers.MaxPool2D((2, 1), (2, 1),padding='same')(X)

        X = residual_block1(res_n_ch[3], X)
        X = self.idenblock(res_n_ch[3], X)
        X = self.idenblock(res_n_ch[3], X)  # 3, 38 ,256
        # X = tf.keras.layers.ZeroPadding2D((1, 1))(X)
        # Xo = tf.keras.layers.AvgPool2D((4, 1), (1, 1))(X)  # 2,40,256
        # Xo = tf.keras.layers.AvgPool2D((2, 1), (1, 1))(Xo)  # 2,40,256
        Xo = tf.reshape(tf.transpose(X, [0, 2, 1, 3]), [-1, tf.shape(X)[2], tf.shape(X)[1]*tf.shape(X)[3]])
        return Xo

