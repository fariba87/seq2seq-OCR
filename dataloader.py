import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy import io
from functools import reduce
import pandas as pd
from utils import *
from ConFig.Config import ConfigReader
#cfg = ConfigReader()
#cfg.batchSize
##############################
#tf.__version__
# print(tf.test.is_gpu_available())



print(tf.config.list_physical_devices('GPU'))
#main_path = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/"
class TrainDatasetReader():
    def __init__(self, main_path = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/" , colnames=['impath', 'label']):
        self.main_path = main_path
        self.dataset_train_paths = {'syntext': 'SynthText_cropped/gt.csv',
                                    'MJsynth': 'mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_train.txt'}#annotation_test.txt'}
        self.colnames = colnames

    def SyntextReader(self):
        '''SynthText_cropped  just contain one annotation file that i considered for train'''
        path = os.path.join(self.main_path, self.dataset_train_paths['syntext'])
        #with f.open(path) as rr:
        # GT_syntext = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/SynthText_cropped/gt.csv"
        #import csv
        #with open(path) as csv_file:
        #    csv = csv.reader(csv_file, delimiter='\t')
        #    for row in csv#_File:
        #        print(row)

        df = pd.read_csv(path, error_bad_lines=False)  # , sep='\t')#, names=colnames  )
        # df.head(4)
        image_and_label_numpy = df.to_numpy().reshape(-1).tolist()
        image_and_label_numpy = list(df.columns.values) + image_and_label_numpy  #first raw
        all_texts = [x.split('\t')[-1] for x in image_and_label_numpy] #6965052
        all_paths = [x.split('\t')[0] for x in image_and_label_numpy]
        # with open(pathGT , encoding = "ISO-8859-1") as f:
        #     while True:
        # line = f.readline()
        all_paths=[os.path.join(self.main_path+'SynthText_cropped', i) for i in all_paths]
        max_len, vocab = maxlen_and_vocab(all_texts) #
        df = create_df(all_paths, all_texts, colnames=self.colnames)
        missing = findnonexistingimages(all_paths)  # []
        data = df.drop(labels=missing, axis=0)
        df = data
        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)
        h = [Hmax]*len(df)
        w = [Wmax]*len(df)
        df['Hmax'] = h
        df['Wmax'] = w
        return df, vocab, max_len #, Hmax, Wmax # xbatch , ybatch

    def MJsynReader(self):

        self.path = os.path.join(self.main_path, self.dataset_train_paths['MJsynth'])
        img_path1 = 'mjsynth/mnt/ramdisk/max/90kDICT32px/'
        img_path=os.path.join(self.main_path, img_path1)
        df = pd.read_csv(self.path, sep=' ', names=self.colnames) #891927

        all_text=df['label'].tolist()  #it is unusued since vocab will be changed!
        all_texts = [df[self.colnames[0]][i].split('_')[-2] for i in range(len(df))]
        all_paths = [os.path.join(img_path, df[self.colnames[0]].tolist()[i][2:]) for i in range(len(df))]
        max_len, vocab = maxlen_and_vocab(all_texts) #23 , 62
        print('maxlen:', max_len)
        print('len_vocab', len(vocab))
        # all_text=[]
        # all_path=[]
        # Wmax=0
        # Hmax=0
        # for i in range(len(df)):
        #     pth = os.path.join(img_path, df[self.colnames[0]].tolist()[i][2:])
        #     print(pth)
        #     if not os.path.exists(pth):#all_paths[i]):
        #         continue
        #     all_text.append(df[self.colnames[0]][i].split('_')[-2])
        #     all_path.append(pth)
        #     img = cv2.imread(pth)
        #     if not type(img)!=np.ndarray: #img == None:
        #         continue
        #     h, w = cv2.imread(pth).shape[:2]
        #
        #     if w>Wmax:
        #         Wmax=w
        #     if h>Hmax:
        #         Hmax=h



        df2 = create_df(all_paths, all_texts, colnames=self.colnames)
        missing = findnonexistingimages(all_paths)  #
        data = df2.drop(labels=missing, axis=0,inplace=False)
        df2 = data
       # max_len, vocab = maxlen_and_vocab(all_texts)
        #df2 = create_df(all_paths, all_texts, colnames=self.colnames)
        all_paths=df2[self.colnames[0]]
        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)
        h = [Hmax]*len(df2)
        w = [Wmax]*len(df2)
        df2['Hmax'] =h#Hmax#h
        df2['Wmax'] =w#Wmax#w
        return df2, vocab, max_len#, Hmax,Wmax

class TestDatasetReader():
    def __init__(self, main_path = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/", is_training=False, colnames=['impath', 'label'], CTC=True, Attn=True):
        self.colnames = colnames

        self.main_path = main_path
        self.is_training = is_training
        self.dataset_paths_test = {'IC03Reader': 'IC03/Word Recognition/TrialTest/word.xml',
                                   'IC13Reader': 'IC13/Word Recognition/Challenge2_Test_Task3_GT.txt',
                                   'IC15Reader': 'IC15/Word Recognition/Challenge4_Test_Task3_GT.txt',
                                   'IIIT5KReader': 'IIIT5K/testCharBound.mat',  # testdata.mat
                                   'SVTReader': 'SVT/svt1/words/test/gt.csv'}  # based on BB

        self.dataset_paths_train = {'IC03Reader': 'IC03/Word Recognition/TrialTrain/word.xml',
                                    'IC13Reader': 'IC13/Word Recognition/Challenge2_Train_Task3_GT.txt',
                                    'IC15Reader': 'IC15/Word Recognition/Challenge4_Train_Task3_GT.txt',
                                    'IIIT5KReader': 'IIIT5K/trainCharBound.mat',  # testdata.mat
                                    'SVTReader': 'SVT/svt1/words/train/gt.csv'}  # based on BB

        self.max_width = -1
        self.max_height = -1
        self.CTC = CTC
        self.Attn = Attn

    def IC03Reader(self):
        # colnames = ['impath', 'label']
        # 'IC03Reader': 'Word Recognition/TrialTest/word.xml'
        #  <image file="word/1/1.jpg" tag="311" />
        # 'Word Recognition/TrialTest/word
        if not self.is_training:
            pathGT = os.path.join(self.main_path, self.dataset_paths_test['IC03Reader'])
            pathImg = self.main_path + 'IC03/Word Recognition/TrialTest'
        else:
            pathGT = os.path.join(self.main_path, self.dataset_paths_traim['IC03Reader'])
            pathImg = self.main_path + 'IC03/Word Recognition/TrialTrain'
        # reading for IC03

        # rootPath = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/IC03/Word Recognition/TrialTest"
        tree = ET.parse(pathGT)
        root = tree.getroot()
        all_paths = []
        all_texts = []
        imgshapes = []
        for imageElement in tqdm(list(root)):
            imageRelativePath = imageElement.get('file')
            img_path = os.path.join(pathImg, imageRelativePath)
            text = imageElement.get("tag")
            img = cv2.imread(img_path)
            if type(img) == None:
                continue
            imgshapes.append(img.shape)
            all_paths.append(img_path)  #1107
            all_texts.append(text)
            # imgtext.append([img_path, text])

        max_len, vocab = maxlen_and_vocab(all_texts) # 14 , 73
        df = create_df(all_paths, all_texts, colnames=self.colnames)
        missing = findnonexistingimages(all_paths)  #[]
        data = df.drop(labels=missing, axis=0)
        df = data
        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)  #589 , 1249
        h = [Hmax]*len(df)
        w = [Wmax]*len(df)
        df['Hmax'] =h
        df['Wmax'] =w
        '''
        image_dir= [os.path.join(dir_img, i) for i in df['img'].values]
'''

# char2int = {ch: i for i, ch in enumerate(vocab)}


        return df, vocab, max_len#, Hmax, Wmax  # ,wmax, hmax #xbatch , ybatch


    def IC13Reader(self):
    # 'IC13Reader': 'Word Recognition/Challenge2_Test_Task3_GT.txt'
    # 'Word Recognition/Challenge2_Test_Task3_Images
    # word_1.png, "Tiredness"
        if not self.is_training:
            pathGT = os.path.join(self.main_path, self.dataset_paths_test['IC13Reader'])
            pathImg = self.main_path + "IC13/Word Recognition/Challenge2_Test_Task3_Images/"

        else:
            pathGT = os.path.join(self.main_path, self.dataset_paths_train['IC13Reader'])
            pathImg = self.main_path + "IC13/Word Recognition/Challenge2_Train_Task3_Images/"

        df = pd.read_csv(pathGT, sep=',', names=self.colnames)  #1095
       # dir_img = os.path.join(self.main_path, dir_)
        all_paths = [os.path.join(pathImg, i) for i in df[self.colnames[0]].values]
        all_texts = df[self.colnames[1]].values

        df = create_df(all_paths, all_texts, colnames=self.colnames)

        missing = findnonexistingimages(all_paths)  # []
        data = df.drop(labels=missing, axis=0)
        df = data
        all_path=df[self.colnames[0]]
        all_texts = df['label'].values
        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)  #1410 , 2155

        h = [Hmax] * len(df)
        w = [Wmax] * len(df)
        df['Hmax'] = h
        df['Wmax'] = w
        max_len, vocab = maxlen_and_vocab(all_texts) # 25 , 77

        '''
        image_dir= [os.path.join(dir_img, i) for i in df['img'].values]
        missing = findnonexistingimages(image_dir)
        data = df.drop(labels=missing, axis=0)'''

        return df, vocab, max_len #,  Hmax, Wmax # xbatch , ybatch


    def IC15Reader(self):
    # 'IC15Reader': 'Word Recognition/Challenge4_Test_Task3_GT.txt'
    # word_1.png, "JOINT"
    # 'Word Recognition/ch4_test_word_images_gt/word_1.png
        if not self.is_training:
            pathGT = os.path.join(self.main_path, self.dataset_paths_test['IC15Reader'])
            pathImg = self.main_path + "IC15/Word Recognition/ch4_test_word_images_gt/"
        else:
            pathGT = os.path.join(self.main_path, self.dataset_paths_train['IC15Reader'])
            pathImg = self.main_path + "IC15/Word Recognition/ch4_train_word_images_gt/"

       # with open(pathGT , encoding = "ISO-8859-1") as f:
       #     while True:
                #line = f.readline()
                #[path,lab]=line.split(',')
        df = pd.read_csv(pathGT,encoding = "ISO-8859-1", sep=',', names=self.colnames) #2074
        all_paths = [os.path.join(pathImg, i) for i in df[self.colnames[0]].values]
        missing = findnonexistingimages(all_paths)  #[]
        data = df.drop(labels=missing, axis=0)
        df= data
        # data is  by itself the df that should be return
        #all_paths = df[self.colnames[0]]
        all_texts = df[self.colnames[1]].tolist()

        df = create_df(all_paths, all_texts, colnames=self.colnames)

        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)  #274, 601

        h = [Hmax] * len(df)
        w = [Wmax] * len(df)
        df['Hmax'] = h
        df['Wmax'] = w
        max_len, vocab = maxlen_and_vocab(all_texts) # 24 78

        return df, vocab, max_len#, Hmax, Wmax  # xbatch , ybatch


    def IIIT5KReader(self):
    # 'IIIT5KReader': 'IIIT5K/testdata.mat'
    # IIIT5K/test/1_1.png
        if not self.is_training:
            pathGT = os.path.join(self.main_path, self.dataset_paths_test['IIIT5KReader'])
        else:
            pathGT = os.path.join(self.main_path, self.dataset_paths_train['IIIT5KReader'])

        GT = io.loadmat(pathGT)
        info = GT['testCharBound'][0]  #3000
        all_paths = []
        all_texts = []
        for i in range(len(info)):
            img_path = info[i][0][0]  # 'test/1002_1.png' , 'PRIVATE'
            text = info[i][1][0]  # 'test/1002_1.png' , 'PRIVATE'
            img_path_full = os.path.join(os.path.join(self.main_path, 'IIIT5K/'), img_path)
            if not os.path.exists(img_path_full):
                continue
            all_paths.append(img_path_full)
            all_texts.append(text)
        # WH
        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)  #2008 , 2576
        max_len, vocab = maxlen_and_vocab(all_texts)  #22 , 62
        df = create_df(all_paths, all_texts, colnames=self.colnames)

        h = [Hmax] * len(df)
        w = [Wmax] * len(df)
        df['Hmax'] = h
        df['Wmax'] = w
        return df, vocab, max_len #,Hmax, Wmax  # , xbatch , ybatch


    def SVTReader(self):
        if not self.is_training:
            pathGT = os.path.join(self.main_path, self.dataset_paths_test['SVTReader'])
            pathImg = self.main_path + 'SVT/svt1/words/test/'
        else:
            pathGT = os.path.join(self.main_path, self.dataset_paths_train['SVTReader'])
            pathImg = self.main_path + 'SVT/svt1/words/train/'

    # 'SVTReader': 'SSVT/svt1/words/test/gt.csv'

    # svt1/words1/test/00_02_340.jpg
    # GT based on full images (it contain BB)
        with open(pathGT, "r") as f:
            lines = f.read().splitlines()
        imgpath=[im.split('\t')[0] for im in lines]
        all_texts=[im.split('\t')[1] for im in lines]
        #df = pd.read_csv(pathGT, sep='\t', names=self.colnames)  #647

        all_paths = [os.path.join(pathImg, i) for i in imgpath]# df[self.colnames[0]].tolist()]
        #all_texts=df[self.colnames[1]].tolist()
        df = create_df(all_paths, all_texts, colnames=self.colnames)
        missing = findnonexistingimages(all_paths)  #[]
        df = df.drop(labels=missing, axis=0)
        #df = data
        # data is  by itself the df that should be return
        all_paths = df[self.colnames[0]].tolist()
        all_texts = df[self.colnames[1]].tolist()


        lis = list(map(hw_img, all_paths))
        Hmax, Wmax = find_max_HW(lis)  # 316 881

        max_len, vocab = maxlen_and_vocab(all_texts) # 13 , 26
    # df = create_df(all_paths, all_texts, colnames=self.colnames)
        h = [Hmax]*len(df)
        w = [Wmax]*len(df)
        df['Hmax'] =h
        df['Wmax'] =w
    # [a.lower() for a in list(vocab)]

        return df, vocab, max_len#, Hmax, Wmax  # , xbatch , ybatch


def create_charset_ds_synth():
    trainds = TrainDatasetReader()
    df1, vc1, mx1 = trainds.MJsynReader()
    print('mx1 is ', mx1)
    voc_train = vc1
    df_train = df1
    all_text_train = df_train['label'].to_numpy().tolist()
    vocab = voc_train
    chars_sorted = sorted(vocab)
    chars_sorted = list(chars_sorted)
    chars_CTC = chars_sorted
    chars_Attn = ["PAD" ,"SOS" ,"EOS"]+chars_sorted #+ ["PAD" ,"SOS" ,"EOS"]
    char2int_CTC = {ch: i for i, ch in enumerate(chars_CTC)}
    char_array_CTC = np.array(chars_CTC)  #black index will be the maximum

    char2int_Attn = {ch: i for i, ch in enumerate(chars_Attn)}
    char_array_Attn = np.array(chars_Attn)  #
    # for tx in text_class_rep:
    #    text_decoded = "".join(char_array_CTC[tx])
    text_class_rep_CTC = []
    text_Attn_train =[]
    for tx in all_text_train:
        text_encoded_CTC = np.array([char2int_CTC[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn =  np.array([char2int_Attn[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn = np.concatenate((np.array([char2int_Attn['SOS']]), text_encoded_Attn, np.array([char2int_Attn['EOS']])))

        text_Attn_train.append(text_encoded_Attn)
        text_class_rep_CTC.append(text_encoded_CTC)
    df_train["CTClabel"] =text_class_rep_CTC
    df_train["Attnlabel"] = text_Attn_train
    return char2int_Attn, char2int_CTC, df_train,mx1

# char2int_Attn, char2int_CTC, df_train , mx1 = create_charset_ds_synth()
# np.save('char2int_CTC_syn.npy', char2int_CTC)
# np.save('char2int_Attn_syn.npy', char2int_Attn)
# df_train.to_csv('df_train_syn.csv')
# np.save('mx1_syn.npy',mx1)
# trainds = TrainDatasetReader()
# df1mj, vc1mj, mx1mj = trainds.MJsynReader() #62 ,  23
# df1mj.to_csv('df1mj.csv')
# np.save('mx1_MJsyn.npy',mx1mj)
# np.save('vcMJsyn.npy',vc1mj)



def create_charset_ds_MJsyn():
    trainds = TrainDatasetReader()
    df1, vc1, mx1 = trainds.MJsynReader()
    print('mx1 is ', mx1)
    voc_train = vc1
    df_train = df1
    all_text_train = df_train['label'].to_numpy().tolist()
    vocab = voc_train
    chars_sorted = sorted(vocab)
    chars_sorted = list(chars_sorted)
    chars_CTC = chars_sorted
    chars_Attn = ["PAD" ,"SOS" ,"EOS"]+chars_sorted #+ ["PAD" ,"SOS" ,"EOS"]
    char2int_CTC = {ch: i for i, ch in enumerate(chars_CTC)}
    char_array_CTC = np.array(chars_CTC)  #black index will be the maximum

    char2int_Attn = {ch: i for i, ch in enumerate(chars_Attn)}
    char_array_Attn = np.array(chars_Attn)  #
    # for tx in text_class_rep:
    #    text_decoded = "".join(char_array_CTC[tx])
    text_class_rep_CTC = []
    text_Attn_train =[]
    for tx in all_text_train:
        text_encoded_CTC = np.array([char2int_CTC[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn =  np.array([char2int_Attn[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn = np.concatenate((np.array([char2int_Attn['SOS']]), text_encoded_Attn, np.array([char2int_Attn['EOS']])))

        text_Attn_train.append(text_encoded_Attn)
        text_class_rep_CTC.append(text_encoded_CTC)
    df_train["CTClabel"] =text_class_rep_CTC
    df_train["Attnlabel"] = text_Attn_train
    return char2int_Attn, char2int_CTC, df_train,mx1

def create_charset_ds_MJsyn_train():
    trainds = TrainDatasetReader()
    df1, vc1, mx1 = trainds.MJsynReader()
    print('mx1 is ', mx1)
    return df1, vc1, mx1,

df_train , vc_train , mx_train = create_charset_ds_MJsyn_train()
# np.save('char2int_CTC_MJ.npy', char2int_CTC)
# np.save('char2int_Attn_MJ.npy', char2int_Attn)
df_train.to_csv('df_train_MJ_true.csv')
vc_train.to_csv('vc_train_MJ_true.csv')
np.save('mx1_MJ_true.npy',mx_train)

# char2int_Attn, char2int_CTC, df_train , mx1 = create_charset_ds_synth()
# np.save('char2int_CTC_MJ.npy', char2int_CTC)
# np.save('char2int_Attn_MJ.npy', char2int_Attn)
# df_train.to_csv('df_train_MJ.csv')
# np.save('mx1_MJ.npy',mx1)
####################################################
def create_charset_ds():#trainds, testds):#TrainDatasetReader, TestDatasetReader):#self, CTC=True, Attn=True):  # take it out of classes for now
    # use all character from train and test?!
    CTC=True

    Attn=True
    trainds = TrainDatasetReader()
    #ds_train = [trainds.SyntextReader() , trainds.MJsynReader()]
    #df_train =[]
    #for i in range(len(ds_train)):
    #    df , vc , mxlen , hmax, wmax= ds_train[i]
    #    df_train.append(df)
    df1, vc1, mx1  = trainds.SyntextReader()
    df2, vc2, mx2 = trainds.MJsynReader()

    voc_train = vc1.union(vc2)
    df_train = pd.concat([df1, df2])
    mx_train = np.max(mx1, mx2)

    testds = TestDatasetReader()
    df1, vc1, mx1  = testds.IC03Reader()
    df2, vc2, mx2  = testds.IC13Reader()

    df3, vc3, mx3  = testds.IC15Reader()
    df4, vc4, mx4  = testds.IIIT5KReader()
    df5, vc5, mx5  = testds.SVTReader()

    voc_test = (vc1.union(vc2)).union(vc3).union(vc4).union(vc5)  #84
    mx_test = np.max((mx1, np.max((mx2, np.max((mx3, np.max((mx4, mx5))))))))  #25
    df_test = pd.concat([df1, df2, df3, df4, df5])  #7923

    all_text_train = df_train['label'].to_numpy().tolist()
    all_text_test  = df_test['label'].to_numpy().tolist()

    just_trainset_for_char = True# False
    if just_trainset_for_char:
        vocab = voc_train
    else:
        vocab = voc_train.union(voc_test)

    chars_sorted = sorted(vocab)
    chars_sorted = list(chars_sorted)

    chars_CTC = chars_sorted
    chars_Attn = ["PAD" ,"SOS" ,"EOS"]+chars_sorted #+ ["PAD" ,"SOS" ,"EOS"]
    char2int_CTC = {ch: i for i, ch in enumerate(chars_CTC)}
    char_array_CTC = np.array(chars_CTC)  #black index will be the maximum

    char2int_Attn = {ch: i for i, ch in enumerate(chars_Attn)}
    char_array_Attn = np.array(chars_Attn)  #
    # for tx in text_class_rep:
    #    text_decoded = "".join(char_array_CTC[tx])
    text_class_rep_CTC = []
    text_Attn_train =[]
    for tx in all_text_train:
        text_encoded_CTC = np.array([char2int_CTC[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn =  np.array([char2int_Attn[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn = np.concatenate((np.array([char2int_Attn['SOS']]), text_encoded_Attn, np.array([char2int_Attn['EOS']])))

        text_Attn_train.append(text_encoded_Attn)
        text_class_rep_CTC.append(text_encoded_CTC)
    df_train["CTClabel"] =text_class_rep_CTC
    df_train["Attnlabel"] = text_Attn_train

    text_class_rep_t_CTC=[]
    text_Attn_test=[]
    for tx in all_text_test:
        text_encoded_CTC = np.array([char2int_CTC[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn = np.array([char2int_Attn[ch] for ch in tx], dtype=np.int32)
        text_encoded_Attn = np.concatenate((np.array([char2int_Attn['SOS']]), text_encoded_Attn, np.array([char2int_Attn['EOS']])))
        text_class_rep_t_CTC.append(text_encoded_CTC)
        text_Attn_test.append(text_encoded_Attn)


    df_test["CTClabel"] = text_class_rep_t_CTC
    df_test["Attnlabel"] = text_Attn_test  #remember if 2last indices: ex : 74:SOS  , 75 EOS (I changed it as 0 , 1, 2)
    #char2int = {ch: i for i, ch in enumerate(chars_sorted)}
    #char_array = np.array(chars_sorted)
    #text_class_rep = []
    #for tx in text:
    #    text_encoded = np.array([char2int[ch] for ch in tx], dtype=np.int32)
    #    text_class_rep.append(text_encoded)
    #for tx in text_class_rep:
    #    text_decoded ="".join(char_array[tx])

# text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32) for encoding
# ''.join(char_array[text_encoded[15:21]]) for decoding

    # i only write it for train. should be repeated for test



        #for tx in text_class_rep:
        #    text_decoded = "".join(char_array[tx])

        #char2int = {ch: i + 2 for i, ch in enumerate(vocab)}
        #char2int['Pad'] = 0
        #char2int['GO'] = 1
        #char2int['EOS'] = 2
    return char2int_Attn, char2int_CTC, df_train, df_test


# based on data generator class, we create generator for train and test dataset
# ##############################################################################
# create tortor by class or function
# class data_generator(tf.keras.utils.Sequence):
#     def __init__(self, df, batch_size = 32, shuffle=True, H=64, CTC=True , Attn=True):
#         self.df = df
#         self.batch_size = batch_size
#         self.H = H
#         self.number_of_samples = len(self.df)
#         self.CTC = CTC
#         self.Attn = Attn
#
#     def __getitem__(self, index):
#         batches = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]  # .iloc[]
#         X, y1,y2 = self.__get_data(batches)  # or y1 and y2
#         return X, y1,y2  # return a complete batch
#
#     def process_image(self, Hmax, Wmax):  # , H, W):  # there is a problem here! since for scale i got 0
#         img = cv2.imread(self.impath)# scale w based on max W in that ds or alllllll?!
#         h, w = img.shape[:2]
#         #img = cv2.resize(img, ())
#        # B = np.int0((w/h)*64)
#        # A= np.int0((Wmax/Hmax)*64)
#        # C= np.int0((A/B)*64)
#         H=64
#         scale = H / img.shape[0]
#         nw = scale * img.shape[1]
#         #img = cv2.resize(img, (nw, H)) / 255.
#         #img = cv2.resize(img, (C, A)) #/ 255.
#         #img = cv2.resize(img, (64, A)) / 255.
#         img = np.array(cv2.resize(img, (np.int0(nw),64))) / 255.
#         h1, w1 = img.shape
#         SeqDivider = 4
#         t = np.ceil(w1/ SeqDivider)
#
#         if np.ndim(img) != 3:
#             img = np.expand_dims(img, axis=-1)
#
#         '''
#         image = tf.keras.preprocessing.image.load_img(path)
#         image_arr = tf.keras.preprocessing.image.img_to_array(image)
#         image_arr = image_arr[ymin:ymin + h, xmin:xmin + w]
#         image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
#         return image_arr / 255.'''
#         return img ,nw,t
#
#     def __get_data(self, batches):
#         colnames = ['impath', 'label'] # two other column (for ctc and attn label) two other column hmax, wmax
#         img_path_list = batches['impath'].to_numpy().tolist()
#         text_list = batches['label'].to_numpy().tolist()
#         Hmax_ds = batches['Hmax'].to_numpy().tolist()
#         Wmax_ds = batches['Wmax'].to_numpy().tolist()
#         text_list_Encoded_CTC   = batches ['CTClabel']
#         text_list_Encoded_Attn  = batches ["Attnlabel"]
#
#         # batches_n = batchs.to_numpy()
#         # img_paths= batches_n[:,0]
#         # labels=batches_n[:,1]
#         w_max = 1
#         max_seq_len = 1
#         max_h = 64
#         x_batch = []
#         y_batch = []
#         times =[]
#         for i in range(self.batch_size):
#             #   im_lab=image_and_label_numpy[i]
#             #  im_path, label=im_lab.split('\t')[:2]
#             self.impath= img_path_list[i]
#             img, w , t = self.process_image( Hmax_ds[i] , Wmax_ds[i] )
#             times.append(t)
#           #  text_list[i]
#           #  label = 'sos' + label + 'eos' #!
#             # need label encoding
#
#             if w > w_max:
#                 w_max = np.int0(w)
#             if len(text_list[i]) > max_seq_len:
#                 max_seq_len = len(text_list[i])  # max_len_per_bacth #text_list_Encoded_CTC
#
#             x_batch.append(img)
#             y_batch.append(np.array([char2int_Attn[j] for j in text_list[i]]))  # it should be encoded one
#         num_channel=3
#         h=64
#         X_batch_resized = np.zeros((self.batch_size, self.H, w_max, num_channel), dtype=np.float32)
#         y_batch_resized_CTC  = np.zeros((self.batch_size, max_seq_len), dtype=np.float32)-1  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
#         y_batch_resized_Attn = np.zeros((self.batch_size, max_seq_len+2), dtype=np.float32)-1  # if zero exist in seq then?! BASED ON  df["attn or ctc"]
#         encoder_mask = np.zeros((self.batch_size, np.int32(np.divide(w_max,16))))
#
#         for i in range(self.batch_size):
#             X_batch_resized[i, :, :x_batch[i].shape[1], :] = x_batch[i]
#             y_batch_resized_CTC[i,  :len(y_batch[i])] = text_list_Encoded_CTC.iloc[i]  # y_batch[i]
#             y_batch_resized_Attn[i, :len(text_list_Encoded_Attn.iloc[i])] = text_list_Encoded_Attn.iloc[i]
#             #encoderInputMask[i][int(sourceImageWidth / self.hparams.sourceEmbeddingDiv):] = 1
#             encoder_mask[i][np.int32(np.divide(x_batch.shape[1],16)):] = 1# np.ones_like((1,x_batch[i].shape[1])
#
#         #if self.CTC :
#          #   return X_batch_resized, y_batch_resized_CTC
#         #else :
#          #   return X_batch_resized, y_batch_resized_Attn  # y_batch_resized  # y_batch_ctc , y_batch_attn
#     #    y_batch_resized_CTC[i, :len(y_batch[i])] = y_batch[i]
#         return X_batch_resized, y_batch_resized_Attn, y_batch_resized_CTC  ,encoder_mask , np.array(times) # y_batch_resized  # y_batch_ctc , y_batch_attn
#
#
#     def _len__(self):
#         # batch size =32
#         # num of samples =len(df)
#         # return the number of batches the generator can produce
#         return np.floor(self.number_of_samples // self.batch_size)
#
#     def __iter__(self):
#         pass
#
#     def next(self):
#         pass
#
#     def __call__(self):
#         for i in range(self.__len__()):
#             yield self.__getitem__(i)

###############################################
# two way to use generator : 1: with tf.data.Dataset 2: DataGenerator
# 1.
#dg = data_generator
#ds = tf.data.Dataset.from_generator(dg,   output_types=3,   output_shapes=tf.string)
#ds = ds.batch(BATCH_SIZE)
#for batch in ds:
 #   x, y = batch

##############################################################################
# 2.
# trainds = TrainDatasetReader()
# testds = TestDatasetReader()
# char2int_Attn, char2int_CTC, df_train, df_test = create_charset_ds()#trainds, testds)
# np.save('char2int_CTC.npy', char2int_CTC)
# np.save('char2int_Attn.npy', char2int_Attn)
# df_train.to_csv('df_train.csv')
# df_test.to_csv('df_test.csv')




#
# batchSize=32
# batch_size = batchSize # or read from config
# traingen_CTC = data_generator(df_train, batch_size=batch_size, CTC =True , Attn=False)# input_size=target_size)
# traingen_Attn = data_generator(df_train, batch_size=batch_size,  CTC =False , Attn=True)# input_size=target_size)
#
# batch_size=32
# valgen_CTC = data_generator(df_test, batch_size=batch_size,  CTC =True , Attn=False)# input_size=target_size)
# valgen_Attn = data_generator(df_test, batch_size=batch_size,  CTC =False , Attn=True)# input_size=target_size)


#model.fit(traingen,
#          validation_data=valgen,
#          epochs=num_epochs)





'''
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([shakespeare_text])
    tokenizer.texts_to_sequences(["First"])
    tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
    max_id = len(tokenizer.word_index)
    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text]))'''
'''
_, text1 = self.SyntextReader()
_, text2 = self.MJsynReader()
S1 = set(list(reduce(lambda x, y: x + y, text1)))
S2 = set(list(reduce(lambda x, y: x + y, text2)))
S=S1.union(S2)'''
