
import cv2
import numpy as np
import tensorflow as tf
from Guided_CTC import model1, model2
targetHeight = 64
from Guided_CTC import chars_Attn
def preprocess(self, image):  # for Attn it is needed to pad the image as input to model
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]
    resized = cv2.resize(im, (int(w * targetHeight / h),targetHeight))
    resized = np.asarray(resized, dtype=np.float32)
    resized = resized / 255.0
    resized = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)  #both channel and batchsize
    return resized


img = cv2.imread('path to test image')
img = preprocess(img)
predict, probs = model1.predict(img)


outputs= model2.predict(img)
outputsClasses = np.squeeze(outputs[0])
probs = np.squeeze(outputs[1])

char_array_Attn = np.array(chars_Attn)
def getString( predictClasses):
    resStr = ""
    for c in predictClasses[0]:
        if c == -1:  #c = intlast  for EOS for Attn, c=-1 for CTC
            break
        resStr += char_array_Attn[c]# class2alphabet[c]
    return resStr
