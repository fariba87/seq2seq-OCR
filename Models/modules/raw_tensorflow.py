import tensorflow as tf
import numpy as np
#X = np.load('../X_batch.npy')
#X = X[0,:32,:320,0]#np.random.random(size=(1, 32, 320))

#XX = X[0,:32,:320,0]#np.random.random(size=(1, 32, 320))
XX = np.random.random((1, 32, 320, 1))
X1 = XX
#X1 = np.expand_dims(XX, axis= 0)

def BatchNorm(input):
    batch_mean, batch_var = tf.nn.moments(input, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    phase_train = True
    mean, var = tf.cond(phase_train,
                    mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
    offset = None
    scale = None
    variance_epsilon =1e-12
    out = tf.nn.batch_normalization(input, mean, var, offset, scale, variance_epsilon, name=None)
    return out

def conv2d_cbr(inputs, filters, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides = [1, stride_size , stride_size, 1], padding = "SAME")
    out = BatchNorm(out)


    return tf.nn.relu(out)####, alpha = leaky_relu_alpha)
def conv2d_crb(inputs, filters, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides = [1, stride_size , stride_size, 1], padding = "SAME")
    out = tf.nn.relu(out)
    out = BatchNorm(out)
    return out#tf.nn.relu(out )####, alpha = leaky_relu_alpha)
def maxpool( inputs, pool_size , stride_size):
	return tf.nn.max_pool2d(inputs, ksize = [1, pool_size, pool_size, 1], padding = "SAME", strides = [1, stride_size, stride_size, 1])
def dense(inputs, units):#weights):
	#x = tf.nn.leaky_relu(tf.matmul(inputs, weights) , alpha = leaky_relu_alpha)
	return tf.compat.v1.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)#tf.nn.dropout

def avg_pooling(x, pool_size=2, stride=2) :
    return tf.layers.average_pooling2d(x, pool_size=pool_size, strides=stride, padding='SAME')

def loss(labels, logits):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def residual_block1(res_n_ch, x):
    a, b, c = res_n_ch
    '''model_res_block = tf.keras.models.Sequential([tf.keras.layers.Conv2D(a, (1, 1)),
                                                  tf.keras.layers.Conv2D(b, (3, 3)),
                                                  tf.keras.layers.Conv2D(c, (1, 1))])'''
    shortcut = x
    #X1 = tf.keras.layers.Conv2D(a, (1, 1),strides =(4,1), padding='same')(x)# valid # i add it myself
    #X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    #X1 = tf.keras.layers.Activation('relu')(X1)
    # filter_height, filter_width, in_channels, out_channels

    X1 = conv2d_cbr(x, [1,1,x.shape[-1], a], [1,4,1,1])

    # X1 = tf.keras.layers.Conv2D(b, (3, 3), strides =(2,1), padding='same')(X1)  # i add it stride
    # X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    # X1 = tf.keras.layers.Activation('relu')(X1)
    X1 = conv2d_cbr(X1, filters =  [3,3,X1.shape[-1], b], stride_size=[1,2,1,1])

    # X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)
    # X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    # X1 = tf.keras.layers.Activation('relu')(X1) ####
    X1 = conv2d_cbr(X1 , filters = [1,1,X1.shape[-1], c] , stride_size = [1,1,1,1])

    # X_shortcut = tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(shortcut)#valid
    # X_shortcut = tf.keras.layers.Activation('relu')(X_shortcut)#####
    # X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X_shortcut = conv2d_crb(X1, [1,1,1,1], stride_size=[1,1,1,1])

#    X1 = tf.keras.layers.Add()([X_shortcut, X1])
    X1 = X_shortcut+X1
    #Xo = tf.keras.layers.Activation('relu')(X1)
    Xo = tf.nn.relu(X1)
    return Xo

def identity_block(res_n_ch, X):
    #  conv_name_base = 'res' + str(stage) + block + '_branch'
    #  bn_name_base = 'bn' + str(stage) + block + '_branch'
    a, b, c = res_n_ch
    shortcut = X
    # X1 = tf.keras.layers.Conv2D(a, (1, 1), padding='same')(X)#
    # X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    # X1 = tf.keras.layers.Activation('relu')(X1)
    X1 = conv2d_cbr(X, [1, 1, X.shape[-1], a], [1, 1, 1, 1])
    # X1 = tf.keras.layers.Conv2D(b, (3, 3),  padding='same')(X1)
    # X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    # X1 = tf.keras.layers.Activation('relu')(X1)
    X1 = conv2d_cbr(X1, filters =  [3,3,X1.shape[-1], b], stride_size=[1,1,1,1])

    # X1 = tf.keras.layers.Conv2D(c, (1, 1), padding='same')(X1)#
    # X1 = tf.keras.layers.Activation('relu')(X1)
    # X1 = tf.keras.layers.BatchNormalization(axis=3)(X1)
    X_shortcut = conv2d_crb(X1, [1,1,1,1], stride_size=[1,1,1,1])
    #
    #X = tf.keras.layers.Add()([shortcut, X1])  # ?
    X2 = X1 + shortcut
    #X = tf.keras.layers.Activation('relu')(X)
    X2 = tf.nn.relu(X2)
    return X2


class ResNet50(tf.keras.Model):#tf.keras.layers.Layer): #
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resblock =residual_block1
        self.idenblock = identity_block
        #self.Conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(10, 8), padding='same') # i add it# raw2-?8
       # self.Conv1 = tf.nn.conv2d(input, [7,7,1,64], [1,10,8,1] , padding = 'same')


    def __call__(self,x):
        res_n_ch = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        #  Xin = tf.keras.layers.Input(shape =(64,160,3))#64,160,3))  # self.input or its shape
        # Xi = tf.keras.layers.ZeroPadding2D((3, 3))(Xin)
        #X = self.Conv1(x )#tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)  #i add the stride
        X = tf.nn.conv2d(x, [7,7,1,64], [1,10,8,1] , padding = "SAME")
        #X = tf.keras.layers.BatchNormalization(axis=3)(X)  #
        X = BatchNorm(X)
        #X = tf.keras.layers.ReLU()(X)  #
        X = tf.nn.relu(X)
        #X = tf.keras.layers.MaxPool2D((6, 3), (2, 4), padding='same')(X)  #3 raw 2-?4


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

model = ResNet50()
logits = model(X1)

# tf.nn.l2_loss
# def model(x):
#
#
#
# import os
# #optimizer = tf.compat.v1.train.Optimizer(10)#use_locking, name)
# class Model(object):
#     def __init__(self):#, hparams: Hparams):
#         # self.char_mapping = {}
#         pass# self.hparams = hparams
#
#     def loss_function(self, real, pred):
#         mask = tf.math.logical_not(tf.math.equal(real, 0))
#         loss_ = self.loss_object(real, pred)
#         mask = tf.cast(mask, dtype=loss_.dtype)
#         loss_ *= mask
#         return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
#
#     def accuracy_function(self, real, pred):
#         accuracies = tf.equal(real, tf.argmax(pred, axis=2))
#         mask = tf.math.logical_not(tf.math.equal(real, 0))
#         accuracies = tf.math.logical_and(mask, accuracies)
#         accuracies = tf.cast(accuracies, dtype=tf.float32)
#         mask = tf.cast(mask, dtype=tf.float32)
#         return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
#     def create_model(self, is_training=True):
#         ### create model
#         self.best_val_acc = 0.0
#         # dataset
#         self.dataLoader = Dataset(self.hparams)
#         self.input_embedding_layer = Resnet50()# mobilenet(is_training=is_training, alpha=0.25)
#         # self.transformer = Transformer(num_layers=self.hparams.num_layers,
#         #                                d_model=self.hparams.d_model,
#         #                                num_heads=self.hparams.num_heads, dff=self.hparams.dff,
#         #                                target_vocab_size=len(self.dataLoader.vocab) + 1, # for PAD add " + 1"
#         #                                maxSourceLength=self.dataLoader.maxSeqLen,
#         #                                maxTargetLength=self.dataLoader.targetMaxLen - 1, rate=self.hparams.dropout_rate)
#         if self.hparams.learning_rate == 'schedule':
#             #self.learning_rate = CustomSchedule(self.hparams.d_model)
#             pass
#         else:
#             self.learning_rate = self.hparams.learning_rate
#         self.optimizer = optimizer = tf.compat.v1.train.Optimizer()#tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#         self.loss_object = loss#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#
#         self.last_epoch = 0
#         self.train_summary_writer = tf.summary.create_file_writer(self.hparams.save_path + '/logs/train')
#         self.valid_summary_writer = tf.summary.create_file_writer(self.hparams.save_path + '/logs/valid')
#         self.checkpoint_dir = os.path.join(self.hparams.save_path, 'train')
#         self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
#         self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
#                                               input_embedding=self.input_embedding_layer,)
#                                               #transformer=self.transformer)
#
#     def load_model(self, model_path=None):
#         if model_path == None:
#             latest = tf.train.latest_checkpoint(self.checkpoint_dir)
#             if latest != None:
#                # logging.info('load model from {}'.format(latest))
#                 self.last_epoch = int(latest.split('-')[-1])
#                 self.checkpoint.restore(latest)
#         else:
#            # logging.info('load model from {}'.format(model_path))
#             self.checkpoint.restore(model_path)
#     def train_step(self, batch_input, batch_target, encoderInputMask):
#         tar_real = np.copy(batch_target[:, 1:])
#         batch_target = batch_target[:, :-1]
#         batch_target[batch_target == self.dataLoader.char_to_idx[self.dataLoader.hparams.end_symbol]] = 0
#         with tf.GradientTape() as tape:
#             input_embeddings = self.input_embedding_layer(batch_input)
#             predictions, attention_weights = self.transformer([input_embeddings, batch_target, encoderInputMask], True)
#             loss = self.loss_function(tar_real, predictions)
#             acc = self.accuracy_function(tar_real, predictions)
#             variables = self.input_embedding_layer.trainable_variables + self.transformer.trainable_variables
#             gradients = tape.gradient(loss, variables)
#             self.optimizer.apply_gradients(zip(gradients, variables))
#         return loss, acc
#     def evaluate(self, batch_input, batch_target, encoderInputMask):
#         encoder_input = self.input_embedding_layer(batch_input)
#         output = np.zeros((batch_input.shape[0], 1))
#         output[:, 0] = self.dataLoader.char_to_idx[self.hparams.start_symbol]
#         for i in range(self.dataLoader.targetMaxLen - 1):
#
#             # predictions.shape == (batch_size, seq_len, vocab_size)
#             predictions, attention_weights = self.transformer([encoder_input, output, encoderInputMask], False)
#
#             # select the last word from the seq_len dimension
#             predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
#
#             predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#
#             output = tf.concat([output, predicted_id], axis=-1)
#         batch_true_char = np.sum(np.equal(output, batch_target))
#         batch_true_str = np.sum(np.prod(np.equal(output, batch_target), axis=1))
#         return batch_true_char, batch_true_str
#
#     def train(self):
#         self.load_model()
#         for epoch in range(self.last_epoch, self.hparams.max_epochs):
#             total_loss = 0
#             # train each batch in dataset
#             for batch in range(100):#int(self.dataLoader.trainTotalData/self.hparams.batch_size)):
#                 batch_input, batch_target, encoderInputMask = self.dataLoader.next_batch_train()
#                 start = datetime.now()
#                 batch_loss, batch_acc = self.train_step(batch_input, batch_target, encoderInputMask)
#                 total_loss += batch_loss
#                 if batch % 1 == 0:
#                     print('Epoch {} Batch {} Loss {:.4f} ACC {:.4f} Time {}'.format(epoch + 1, batch, batch_loss,
#                                                                                     batch_acc, datetime.now()-start))
#
#             # evaluate on train set
#             logging.info('evaluate on train set')
#             cnt_true_char = 0
#             cnt_true_str = 0
#             sum_char = 0
#             sum_str = 0
#             for batch in tqdm(range(30)):#range(int(self.dataLoader.trainTotalData/self.hparams.batch_size)):
#                 batch_input, batch_target, encoderInputMask = self.dataLoader.next_batch_train()
#                 batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target, encoderInputMask)
#                 cnt_true_char += batch_true_char
#                 cnt_true_str  += batch_true_str
#                 sum_char += batch_input.shape[0] * self.dataLoader.targetMaxLen
#                 sum_str  += batch_input.shape[0]
#             train_char_acc = cnt_true_char/sum_char
#             train_str_acc  = cnt_true_str/sum_str
#
#             # evaluate on valid set
#             logging.info('evaluate on valid set')
#             cnt_true_char = 0
#             cnt_true_str = 0
#             sum_char = 0
#             sum_str = 0
#             for batch in tqdm(range(30)):#range(int(self.dataLoader.validTotalData/self.hparams.batch_size)):
#                 batch_input, batch_target, encoderInputMask = self.dataLoader.next_batch_valid()
#                 batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target, encoderInputMask)
#                 cnt_true_char += batch_true_char
#                 cnt_true_str  += batch_true_str
#                 sum_char += batch_input.shape[0] * self.dataLoader.targetMaxLen
#                 sum_str  += batch_input.shape[0]
#             valid_char_acc = cnt_true_char/sum_char
#             valid_str_acc  = cnt_true_str/sum_str
#             # save checkpoint
#
#             if self.hparams.save_best:
#                 if self.best_val_acc < valid_str_acc:
#                     self.checkpoint.save(file_prefix = self.checkpoint_prefix)
#             else:
#                 self.checkpoint.save(file_prefix = self.checkpoint_prefix)
#
#             # write log
#             with self.train_summary_writer.as_default():
#                 tf.summary.scalar('loss', total_loss, step=epoch)
#                 tf.summary.scalar('character accuracy', train_char_acc, step=epoch)
#                 tf.summary.scalar('sequence accuracy', train_str_acc, step=epoch)
#
#             with self.valid_summary_writer.as_default():
#                 tf.summary.scalar('character accuracy', valid_char_acc, step=epoch)
#                 tf.summary.scalar('sequence accuracy', valid_str_acc, step=epoch)
#
#             # log traing result of each epoch
#             logging.info('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
#             logging.info('Accuracy on train set:')
#             logging.info('character accuracy: {:.6f}'.format(train_char_acc))
#             logging.info('sequence accuracy : {:.6f}'.format(train_str_acc))
#             logging.info('Accuracy on valid set:')
#             logging.info('character accuracy: {:.6f}'.format(valid_char_acc))
#             logging.info('sequence accuracy : {:.6f}'.format(valid_str_acc))




