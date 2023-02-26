import tensorflow as tf
import numpy as np
def residual_block1(res_n_ch, x):
    a, b, c = res_n_ch
    '''model_res_block = tf.keras.models.Sequential([tf.keras.layers.Conv2D(a, (1, 1)),
                                                  tf.keras.layers.Conv2D(b, (3, 3)),
                                                  tf.keras.layers.Conv2D(c, (1, 1))])'''
    shortcut = x
    X1 = tf.keras.layers.Conv2D(a, (1, 1), padding='valid')(x)
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(b, (3, 3), padding='same')(X1)
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X1 = tf.keras.layers.Activation('relu')(X1)

    X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)
    X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X_shortcut = tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 1), padding='valid')(shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X1 = tf.keras.layers.Add()([X_shortcut, X1])
    Xo = tf.keras.layers.Activation('relu')(X1)
    return Xo

def identity_block(res_n_ch, X):
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
    X = tf.keras.layers.Add()([shortcut, X])  # ?
    X = tf.keras.layers.Activation('relu')(X)
    return X

def ResNet_backbone( Xin):
    res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
    Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(Xi)  #
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
    X = tf.keras.layers.ZeroPadding2D((1, 1))(X)
    Xo = tf.keras.layers.AvgPool2D((4, 1), (1, 1))(X)  # 2,40,256
    Xo = tf.keras.layers.AvgPool2D((2, 1), (1, 1))(Xo)  # 2,40,256

    Xo = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 1))(
        Xo)  # 40 ,256 # or use feature_map_to_featurevec function
    #resnet_model = tf.keras.models.Model(inputs=Xin, outputs=Xo)
    #resnet_model.summary()
    return Xo


def attential_guidance(X, y , training=True):
    # input :feature vector  [None ,40,256 ]
    # output char sequence
    # based on GRU
    #tf.keras.layers.GRU
    #tf.keras.layers.GRU
    based_on_keras_attention =None
    if based_on_keras_attention:
        from attnkeras import attention_with_keras
        encoder_in = X
        decoder_in = y
        decoder_out= y
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
class Attention_model()
    def __init__(self):
        self.feature_extraction = ResNet_backbone
        self.Attention
    def __call__(self, x1):
        x , y ,weight = x1
        feat_vec = self.feature_extraction(x)




