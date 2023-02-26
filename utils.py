# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy import io
from functools import reduce
import numpy as np
import tensorflow as tf
import pandas as pd
from ConFig.Config import ConfigReader
cfg = ConfigReader()
################################
def findnonexistingimages(img_list):
    isExist = [os.path.exists(path) for path in img_list]  # all paths
    missing_indices_in_dataset = [i for i, x in enumerate(isExist) if x == False]
    # indices for images not existing in data directory (338)
    return missing_indices_in_dataset

def maxlen_and_vocab(all_texts):
    max_len = len(sorted(all_texts, key=lambda x: len(x))[-1])
    vocab = set(list(reduce(lambda x, y: x + y, all_texts)))  # all char in texts
    return max_len,vocab
def create_df(all_paths,all_texts, colnames = ['impath', 'label']):
    imcol = pd.Series(all_paths, name=colnames[0])
    texcol = pd.Series(all_texts, name=colnames[1])
    df = pd.concat([imcol, texcol], axis=1)
    return df
def find_max_HW(b):
    H =sorted(b)[-1][-1]
    W =sorted(b, key=lambda a: a[-1])[-1][-1]
    return H,W


def hw_img(path):
    if path =='/media/Archive4TB3/Data/textImages/EN_Benchmarks/mjsynth/mnt/ramdisk/max/90kDICT32px/2911/6/77_heretical_35885.jpg':
        h=0
        w=0
    elif (os.path.exists(path)) & (type(cv2.imread(path))==np.ndarray):
        h, w = cv2.imread(path).shape[:2]

    else:
        h=0
        w=0
    return (h, w)
######################################################################

def preprocess(image_path, Height):
    image=cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    img_resized = cv2.resize(img, (int(w * Height / h), Height))
    res_norm = (np.asarray(img_resized, dtype=np.float))/255.0
    res_norm_b = np.expand_dims(np.expand_dims(res_norm, axis=0), axis=-1) #(1,h,w,1)
    return res_norm_b #resize,normalize, bs extended
BATCH_SIZE= 64




def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def create_padding_mask(seq, encoderInputMask=None):
    if encoderInputMask is None:
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    else:
        return encoderInputMask[:, np.newaxis, np.newaxis, :]


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask
# (seq_len, seq_len)
def convertbacktonumpy(A):  #AttnLabel column of DF was save as string
    AA = A[1:-1]
    return np.fromstring(AA, dtype=int, sep=' ')


def paddedseq(label, maxlen, defal =0):
    len_label = len(label)
    padsize = maxlen - len_label
    #    paddings = [[0, padsize]]# , [0, padsize], [0, 0]]
    #    paddedseq= tf.pad(label, paddings, "CONSTANT", constant_values=-1)
    #    paddedseq = np.concatenate([label , np.zeros(padsize)-1])
    paddedseq = label.tolist() + [defal] * padsize  # 0 or -1 : here we use 0 padding
    return np.array(paddedseq)

def encode_txt2ind(x, chars):
    if 'SOS' in chars:
        text_encoded = np.concatenate((np.expand_dims(chars['SOS'], 0),
                                   np.array([chars[ch] for ch in str(x)], dtype=np.int32),
                                   np.expand_dims(chars['EOS'], axis=0)), axis=0)
    else:
        text_encoded = np.array([chars[ch] for ch in str(x)], dtype=np.int32)
    return text_encoded

def create_in_out_decoder(yAtt, lenvoc):
    #lenvoc=len(vocab)
    yin = yAtt
    tar_real = np.copy(yin[:, 1:])
    yin = yin[:, :-1]
    yin[yin == lenvoc+2] = 0
    return yin,tar_real