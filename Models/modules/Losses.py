import tensorflow as tf

def CTCLoss(y_true, y_pred, times):
    # Compute the training-time loss value
 #   idx = tf.where(tf.not_equal(y_true, -1))
  #  targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))
 #   ytrue = targets
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    #label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    label_length = tf.cast(times, dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def loss_and_acc_function(real, pred, times_in):
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
    label_len = tf.shape(real)[-1]
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
                          logits_time_major=False, # if false: [BS, seqlen, classes]
                          blank_index=77)#char2int_Attn_array.shape[0])  # (None,)
    # pred = tf.transpose(pred , [1,0,2])
    # #loss = tf.compat.v1.nn.ctc_loss(targets, pred, times_in1, ignore_longer_outputs_than_inputs=True)
    # loss = tf.compat.v1.nn.ctc_loss(targets, pred, times_in1, ignore_longer_outputs_than_inputs=True)
    loss = tf.reduce_mean(loss)
    logits = tf.transpose(pred, (1, 0, 2))
    top_paths=30
    decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(logits,
                                                                times_in1,
                                                                merge_repeated=False,
                                                                top_paths=top_paths)
    ctc_prediction_len = 30
    batchSize = 1
    # res_lab = tf.compat.v1.sparse_to_dense(tf.cast(decoded[0].indices, tf.int32),
    #                              tf.stack([batchSize, ctc_prediction_len]),
    #                              decoded[0].values,
    #                              default_value=-1)
    acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    return loss , acc

def loss_and_acc_function_DL(real, pred):
    BS = tf.shape(real)[0]#self.labels_CTC)[0]
    label_len = tf.shape(real)[-1]
    idx = tf.where(tf.not_equal(real, -1))
    targets = tf.SparseTensor(idx, tf.gather_nd(real, idx), tf.cast(tf.shape(real), tf.int64))
    input_length = tf.cast(tf.fill((BS,), pred.shape[1]),tf.int32)
    times_in1 = tf.cast(tf.fill((BS,),25), tf.int32)
  #  times_in1 = tf.cast(tf.fill((BS,),pred.shape[1]), tf.int32)
    loss = tf.nn.ctc_loss(labels=tf.cast(targets, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
                          logits=pred,
                          label_length=None,#times_in,# tf.cast(BS, tf.int32),
                          logit_length=times_in1,  # tf.cast(tf.shape(pred2)[1],tf.int32),
                          logits_time_major=False, # if false: [BS, seqlen, classes]
                          blank_index=77)#char2int_Attn_array.shape[0])  # (None,)

    loss = tf.reduce_mean(loss)
    return loss #, acc

import numpy as np
from DataLoad.CTC_data import vocab
characters = list(vocab)#[x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def decode_batch_predictions(pred, real):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    #acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(results, tf.int32), real))
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def loss_function_transformer(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    #print(tf.reduce_sum(loss_) / tf.reduce_sum(mask))
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def lossCTC(y_true , y_pred,i1,l1, lenvoc):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    i11 = tf.cast(i1, dtype="int64")
    l11 = tf.cast(l1, dtype="int64")

  #  loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)#, ignore_longer_outputs_than_inputs=True)#batch_cost
  #   loss = tf.compat.v1.nn.ctc_loss(labels=tf.cast(y_true, tf.int32),  # tf.cast(self.labels_CTC, tf.int32),
  #                         logits=y_pred,
  #                         label_length=l1,#label_length,#None,#times_in,# tf.cast(BS, tf.int32),
  #                         logit_length=i1,#input_length,#times_in1,  # tf.cast(tf.shape(pred2)[1],tf.int32),
  #                         logits_time_major=False, # if false: [BS, seqlen, classes]
  #                         ignore_longer_outputs_than_inputs =True)
  #                         #blank_index=77)#char2int_Attn_array.shape[0])  # (None,)
    idx = tf.where(tf.not_equal(y_true, lenvoc))
    targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))

    '''loss = tf.compat.v1.nn.ctc_loss(labels = tf.cast(targets, tf.int32),
    inputs=y_pred,
    sequence_length=tf.cast(i11, tf.int32),
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=True,
    time_major=False,
    logits=None)'''
    # input_length = tf.expand_dims(input_length, -1)
    # label_length = tf.expand_dims(label_length, -1)
    #i11 = y_pred.shape[1]
    input_length = i11 *tf.ones(shape=(batch_len, 1), dtype="int64")#*i1#input_length *
    label_length = l11*tf.ones(shape=(batch_len, 1), dtype="int64")#*l1#label_length *
    #label_length =tf.expand_dims(l11, axis=-1)
    #input_length = tf.expand_dims(i11, axis=-1)
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)#input_length,label_length)#input_length, label_length)#, ignore_longer_outputs_than_inputs=True)#batch_cost

    # decoded = tf.keras.backend.ctc_decode(y_pred, input_length=input_length, greedy=True)
   # acc= float(decoded[1][0][0])
    #label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

    return loss#, acc

def acc_function_CTC(y_true, y_pred ):
     results= tf.keras.backend.get_value(tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                             greedy=True)[0][0])

     idx = tf.where(tf.not_equal(y_true, 96))
     targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))
     idx = tf.where(tf.not_equal(results, -1))
     prediction = tf.SparseTensor(idx, tf.gather_nd(results, idx), tf.cast(tf.shape(results), tf.int64))

     acc = 1.0  -tf.reduce_mean(tf.edit_distance(tf.cast(prediction, tf.int32), tf.cast(targets, tf.int32))).numpy()
     # i = 0
     # for x in results:
     #     print("original_text =  ", test_orig_txt[i])
     #     print("predicted text = ", end='')
     #     for p in x:
     #         if int(p) != -1:
     #             print(char_list[int(p)], end='')
     #     print('\n')
     #     i += 1

     #GT =

     return acc

def accuracy_function_transformer(real, pred):
  real = tf.cast(real, dtype=tf.int64)
  pred = tf.cast(pred, dtype=tf.int64)
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class AccuracyCTC(tf.keras.losses.Loss):
    def __init__(self,lenvocab=0):#, input_length =1.0, lenvocab=0, **kwargs ):
        #self.input_length = input_length
        self.lenvocab = lenvocab
        super(AccuracyCTC, self).__init__()#**kwargs)
    def __call__(self, y_true, y_pred):
       # print(np.ones(y_pred.shape[0]) * y_pred.shape[1].shape)
       # print(y_pred.shape)

       # print()
        results = tf.keras.backend.get_value(
            tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                                        greedy=True)[0][0])

        idx = tf.where(tf.not_equal(y_true, self.lenvocab))
        targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))
        idx = tf.where(tf.not_equal(results, -1))#
        prediction = tf.SparseTensor(idx, tf.gather_nd(results, idx), tf.cast(tf.shape(results), tf.int64))

        acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(prediction, tf.int32), tf.cast(targets, tf.int32))).numpy()
        return acc
    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "threshold": self.threshold}
#model.compile(metrics = [AccuracyCTC(input_length=1)])
class CTCMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
      #  self.threshold = threshold
      #  self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        #metric = self.huber_fn(y_true, y_pred)
        results = tf.keras.backend.get_value(
            tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                                        greedy=True)[0][0])

        idx = tf.where(tf.not_equal(y_true, self.lenvocab))
        targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))
        idx = tf.where(tf.not_equal(results, -1))
        prediction = tf.SparseTensor(idx, tf.gather_nd(results, idx), tf.cast(tf.shape(results), tf.int64))

        acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(prediction, tf.int32), tf.cast(targets, tf.int32))).numpy()

        self.total.assign_add(tf.reduce_sum(acc))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "threshold": self.threshold}


def create_acc_fn(lenvocab=0):
    def acc_fn(y_true, y_pred):
        results = tf.keras.backend.get_value(
            tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                                        greedy=True)[0][0])

        idx = tf.where(tf.not_equal(y_true, lenvocab))
        targets = tf.SparseTensor(idx, tf.gather_nd(y_true, idx), tf.cast(tf.shape(y_true), tf.int64))
        idx = tf.where(tf.not_equal(results, -1))
        prediction = tf.SparseTensor(idx, tf.gather_nd(results, idx), tf.cast(tf.shape(results), tf.int64))

        acc = 1.0 - tf.reduce_mean(tf.edit_distance(tf.cast(prediction, tf.int32), tf.cast(targets, tf.int32))).numpy()
        return acc
        #
        # self.total.assign_add(tf.reduce_sum(acc))
        # self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
