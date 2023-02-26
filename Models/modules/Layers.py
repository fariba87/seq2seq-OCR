import tensorflow as tf
import numpy as np
from Models.modules.utils import positional_encoding, scaled_dot_product_attention, create_look_ahead_mask, create_padding_mask

#@tf.function
def Adj_mat(n):
    beta = 0.5
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i:] = A[i:, i] = np.arange(n - i)
    A = tf.convert_to_tensor(A)
    A_adj = 1 - tf.math.sigmoid(A + beta)
    return A_adj

#@tf.function
def Adj_mat_tf(n):
    beta = 0.5
   # A = tf.zeros((n, n))
    # for i in range(n):
    A = tf.gather_nd(indices =[tf.range(n)],#[i] for i in tf.range(n)

                # params=[[tf.range(n-i)] for i in tf.range(n).numpy()])
                 params =[[tf.concat([tf.cast(tf.zeros(i), tf.int64), tf.range(n-i)], axis=0)]
                          for i in tf.range(n)])
    B =  tf.gather_nd(indices =[tf.range(n)], #.numpy()[i] for i in tf.range(n)
                      params=[[tf.concat([ tf.range(i , limit = -1,delta = -1),tf.cast(tf.zeros(n-i-1), tf.int32)], axis=0)]
                             for i in tf.range(n)])
    #     A[i, i:] = A[i:, i] = tf.range(n - i)
    sum = tf.cast(A, tf.int64) + tf.cast(B, tf.int64)
    #Asum = tf.convert_to_tensor(sum)
    Asum = tf.squeeze(sum, axis=1)
    A_adj = 1 - tf.math.sigmoid(tf.math.add(tf.cast(Asum, tf.float32),tf.constant(beta, dtype =tf.float32)))#Asum + tf.convert_to_tensor(beta))
    return A_adj

max_seq_length = 300
#Adj_mat_list = list(map(Adj_mat, range(max_seq_length)))
def adjacency_matrix_list(max_seq_length =300):
    return list(map(Adj_mat, range(max_seq_length)))
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

    # def build(self, input_shape):  # what is input shape?
    #     self.W = self.add_weight(
    #         shape=(input_shape[2], self.units), #unit is N in defination of W , input_shape [BS, T,Q]
    #         dtype=self.dtype,
    #         initializer='glorot_uniform',
    #         regularizer=tf.keras.regularizers.l2(0.01))


    def sim_mat(self):
        h = self.Hmatrix  #[None, 40 ,1024[

        # n = h.shape.as_list()[-1]
        # h = tf.keras.layers.Dense(n)(h)
        # L2Norm = tf.sqrt(tf.reduce_sum(tf.pow(h, 2), axis=-1))
        # h1 = h / tf.tile(tf.expand_dims(L2Norm, axis=-1), [1, 1, n])#sh[-1]])
        # A_S = tf.cast(tf.matmul(h1, tf.transpose(h, [0, 2, 1])),tf.float32)
        A_S = tf.keras.layers.Dot(axes=2, normalize=True)([h, h])
        #A_S = tf.matmul(H, tf.transpose(H),) / (tf.pow(tf.math.l2_normalize(H), 2))
        #A_S = tf.matmul(tf.transpose(h, perm=[0, 2, 1]), h) / (tf.pow(tf.tf.math.l2_normalize(h), 2))
        #self.A_S = A_S
        return A_S   # [none , 40, 40]

    def __call__(self, X, GCNin):  #in calling :both X and resnet.output
        self.X = X  #[BS, T, Q]
        self.Hmatrix = GCNin#[0,:,:]
        n = GCNin.shape.as_list()[-2]
        #n = tf.shape(GCNin)[-2]



        #n= tf.shape(GCNin)[-2]#.GCNin.get_shape()[-2]#tf.shape(GCNin)[-2].numpy()
        bs = tf.shape(GCNin)[0]

        #bs=32
        A_D = tf.tile(tf.expand_dims(Adj_mat(n),axis=0),[bs,1,1])#Adj_mat_list[n]
        A_D = tf.cast(A_D, dtype= tf.float32) #how can i resize?
        # print(A_D.shape)
        #A_D = tf.keras.layers.Lambda(lambda: Adj_mat_list[40])()


        # X = tf.nn.dropout(X, self.rate)
        #self.A = tf.matmul (self.sim_mat()*self.Adj_mat())
        #n = self.X.get_shape().as_list()[-2]
      #  self.A = tf.matmul(self.sim_mat(),A_D)#dj_mat_list[n])#self.A_adj)
        self.A  =self.sim_mat()@A_D
        X = self.A @ self.X# @ self.W
       # h = tf.matmul(h @ h.T) / tf.math.l2_normalize(h @ h.T)
        return X #self.activation(X)


def BiLSTM_func(X, times):
    #max_len = tf.shape(X)[1]
    max_len = X.get_shape()[1]
    mask = tf.transpose((tf.sequence_mask(times, maxlen=max_len, dtype=tf.float32, name=None)), perm=[0,2,1])  # tf.reduce_max(self.times)
    #print(X.shape)
    #print(mask.shape)
    # # mask should be of size [None, seqlen, 1]
    X = X * mask
    X = tf.keras.layers.Masking(mask_value=0.0)(X)
    X1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(X)
    return X1


def BiLSTM_func1(X, times):
    #max_len = tf.shape(X)[1]
    max_len = X.get_shape()[1]
    mask = tf.transpose((tf.sequence_mask(times, maxlen=max_len, dtype=tf.float32, name=None)), perm=[0,2,1])  # tf.reduce_max(self.times)
    #print(X.shape)
    #print(mask.shape)
    # # mask should be of size [None, seqlen, 1]
    X = X * mask
    X = tf.keras.layers.Masking(mask_value=0.0)(X)
    X1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,  activation='tanh'))(X)
    return X1


class CTCLayer_DL(tf.keras.layers.Layer):

    def __init__(self, name=None):

        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)  #[none , 1]
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        x = tf.keras.layers.Dense(512)(x)  # i add it myself
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, maxSourceLength, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maxSourceLength, self.d_model)  # i add it myself (maxSourceLength)6144

        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
#        self.mult = tf.keras.layers.Lambda(lambda x : x* 22.63)# tf.math.sqrt(tf.cast(self.d_model, tf.float32)))#Multiply()
#        self.add = tf.keras.layers.Add()


    def call(self, x, training, mask):

        seq_len=tf.shape(x)[1]# x.get_shape()[1]#
 #       x = self.mult(x)#([x, tf.math.sqrt(tf.cast(self.d_model, tf.float32))])
#        x = self.add([x, self.pos_encoding[:, :seq_len, :] ])
        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        #x = tf.keras.layers.multiply()([x, tf.math.sqrt(tf.cast(self.d_model, tf.float32))])#(x)
        #x = tf.keras.layers.add([x, self.pos_encoding[:, :seq_len, :]])
        x = tf.keras.layers.Dense(self.d_model)(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #c = tf.transpose(self.pos_encoding,[0,2,1])  # i add it myself
        x += self.pos_encoding[:, :seq_len, :]#c[:,:seq_len,:]# self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, maxTargetLength, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maxTargetLength, d_model) # 62[1 ,maxTargetLength, d_model]maxTargetLength

        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads, dff, target_vocab_size, maxSourceLength, maxTargetLength, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, maxSourceLength=maxSourceLength,
                               rate=rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               target_vocab_size=target_vocab_size, maxTargetLength=maxTargetLength, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar, encoderInputMask = inputs
        #encoderInputMask = tf.keras.layers.Dense(101)(encoderInputMask)
        encoderInputMask=  encoderInputMask[:, :-2]# 2
        encPadding_mask, look_ahead_mask = self.create_masks(inp, tar, encoderInputMask, training)
        #none, 1,1,25  - none, 1,27,27
        enc_output = self.encoder(inp, training, encPadding_mask)  # (batch_size, inp_seq_len, d_model)
        # 1, 101,512
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, encPadding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, inp, tar, encoderInputMask, training):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
       # tar = tf.expand_dims(tar, axis= 0)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) #27*27
        dec_target_padding_mask = create_padding_mask(tar) #None,1,1,27
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        if training:
            # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
            padding_mask = create_padding_mask(inp, encoderInputMask) #none,1,1,25
        else:
            padding_mask = None
        return padding_mask, look_ahead_mask


# class GraphConvolutionLayer(tf.keras.layers.Layer):
#     # input X :[BS, T, Q]
#     # A_S :[T,T]
#     # A_D :[T,T]
#     # H : [T,Q]
#     # W : [Q, N]
#     # output: BS*T*N
#     def __init__(self, units, activation=tf.identity):  # A
#         super(GraphConvolutionLayer, self).__init__()
#
#         self.activation = activation
#         self.units = units  # what is unites? =N
#         # self.A = A it can also be defined for constant sequence length for each batch
#
#     def build(self, input_shape):  # what is input shape?
#         self.W = self.add_weight(
#             shape=(input_shape[2], self.units),  # unit is N in defination of W , input_shape [BS, T,Q]
#             dtype=self.dtype,
#             initializer='glorot_uniform')#,
#            # regularizer=tf.keras.regularizers.l2(0.01))
#
#     def sim_mat(self):
#         h = self.Hmatrix  # [None, 40 ,1024[
#
#         n = h.shape.as_list()[-1]
#         h = tf.keras.layers.Dense(n)(h)
#         L2Norm = tf.sqrt(tf.reduce_sum(tf.pow(h, 2), axis=-1))
#         h1 = h / tf.tile(tf.expand_dims(L2Norm, axis=-1), [1, 1, n])  # sh[-1]])
#         A_S = tf.cast(tf.matmul(h1, tf.transpose(h, [0, 2, 1])), tf.float32)
#         A_S = tf.keras.layers.Dot(axes=2, normalize=True)([h, h])
#
#         # A_S = tf.matmul(H, tf.transpose(H),) / (tf.pow(tf.math.l2_normalize(H), 2))
#         # A_S = tf.matmul(tf.transpose(h, perm=[0, 2, 1]), h) / (tf.pow(tf.tf.math.l2_normalize(h), 2))
#         # self.A_S = A_S
#         return A_S  # [none , 40, 40]
#
#     def call(self, X):#, GCNin):  # in calling :both X and resnet.output
#         self.X = X  # [BS, T, Q]
#         self.Hmatrix = X#GCNin  # [0,:,:]
#         n = X.shape.as_list()[-2]  # 40
#         bs = tf.shape(X)[0]
#         # bs=32
#         A_D = tf.tile(tf.expand_dims(Adj_mat_list[n], axis=0), [bs, 1, 1])
#         A_D = tf.cast(A_D, dtype=tf.float32)  # how can i resize?
#         #print(A_D.shape)
#         # A_D = tf.keras.layers.Lambda(lambda: Adj_mat_list[40])()
#
#         # X = tf.nn.dropout(X, self.rate)
#         # self.A = tf.matmul (self.sim_mat()*self.Adj_mat())
#         # n = self.X.get_shape().as_list()[-2]
#         #  self.A = tf.matmul(self.sim_mat(),A_D)#dj_mat_list[n])#self.A_adj)
#         self.A = self.sim_mat() @ A_D
#         X = self.A @ self.X @ self.W
#         # h = tf.matmul(h @ h.T) / tf.math.l2_normalize(h @ h.T)
#         return X  # self.activation(X)

class ADJlayer(tf.keras.layers.Layer):
    def __init_(self):
        super(ADJlayer, self).__init__()

    def call(self,X):
        n = tf.shape(X)[-2]  #-2 bood
        bs = tf.shape(X)[0]
        beta = 0.8
        #n = tf.shape(X)[-2]
        i = tf.constant(2, dtype=tf.int32)
        # n = tf.constant(10, dtype=tf.int32)
        A = tf.expand_dims(tf.range(n, dtype=tf.int32), axis=0)

        c = lambda i, n, A: tf.less_equal(i, n)
        b = lambda i, n, A: (tf.add(i, 1),
                             n,
                             tf.concat([A, [
                                 tf.concat([tf.zeros(i, dtype=tf.int32), tf.range(1, n - i + 1, dtype=tf.int32)],
                                           axis=0)]],
                                       axis=0)
                             )  # tf.add(i, 1)
        i, n, A = tf.while_loop(cond=c, body=b, loop_vars=[i, n, A],#,
                                shape_invariants=[i.get_shape(), n.get_shape(), tf.TensorShape([None, None])])#A.set_shape([None,None])])#tf.TensorShape([None, n])])
        A = tf.add(A, tf.transpose(A))
        A_adj = 1 - tf.math.sigmoid(tf.math.add(tf.cast(A, tf.float32), tf.constant(beta, dtype=tf.float32)))
        A = tf.reshape(A, [1, tf.shape(A_adj)[0], tf.shape(A_adj)[1]])
        A_adj_broadcast = tf.tile(A, [bs,1,1])

        #input1 = tf.keras.layers.Input(tf.shape(A_adj))#.shape.as_list())#(67, 67))
        #A_adj_broadcast = tf.broadcast_to(A_adj, [tf.keras.backend.shape(input1)[0]]+tf.shape(A_adj))#A_adj.shape.as_list())# (67,67)])
        #A_adj_broadcast = tf.broadcast_to(A_adj, [tf.shape(input1)[0]] + tf.shape(A_adj))
        return  A_adj_broadcast



    # Shape of const is now (None, 4)