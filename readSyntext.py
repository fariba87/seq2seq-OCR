import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy import io
import re
from functools import reduce
import pandas as pd
#tf.__version__
# print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))


class SynthTextData():
    def __init__(self):
        self.main_path = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/SynthText"
        self.gt_path = os.path.join(self.main_path, "gt.mat")
        self.num_images_exist = self.count_images()
        self.num_images = self.count_all_syntext_images()
        self.max_width = -1
        self.max_height = -1

    def count_all_syntext_images(self):
        GT_syntext = io.loadmat(self.gt_path)
        num_images = len(GT_syntext['txt'][0])
        return num_images

    def count_images(self):
        list_syntext = os.listdir(self.main_path)
        list_clean_syntext = [path for path in list_syntext if path[-4:] not in ['.mat', 'tore']]
        num_folder_syn = len(list_clean_syntext)  # 200
        print(num_folder_syn)
        # if f[-4:] in ['.mat' , 'stor']
        #####################################################
        num_all_images_exist = 0
        for path in list_clean_syntext:
            subdir = os.path.join(self.main_path, path)
            subdir_imgs = os.listdir(subdir)
            lendir = len(subdir_imgs)
            num_all_images_exist += lendir
            # print('number of images in folder {} is {}'.format(path, lendir))
        print('counting all images in all folders(overall){}:'.format(num_all_images_exist))
        return num_all_images_exist


        def GetData(self):
            GT_syntext = io.loadmat(self.gt_path)
            all_text = []
            all_text_image = []
            D = []
            tex = []
            A = []
            info_ = []
            coor = []
            MM = []
            for i in range(self.num_images):
                img_path = os.path.join(self.images_path, GT_syntext['imnames'][0][i][0])
                isexist = os.path.exists(img_path)
                if isexist:
                    texts_per_image = GT_syntext['txt'][0][i]
                    words = re.sub(' +', ' ', " ".join(texts_per_image.tolist()).replace("\n", " ").strip()).split(" ")
                    dim = len(GT_syntext['wordBB'][0][i].shape)
                    if dim == 3:
                        _, _, num = GT_syntext['wordBB'][0][i].shape
                    else:
                        num = 1
                        GT_syntext['wordBB'][0][i] = np.expand_dims(GT_syntext['wordBB'][0][i], axis=-1)

                    coor = []
                    MM = []
                    H = []
                    W = []
                    tex = []

                    for kk in range(num):
                        text = words[kk]
                        tex.append(text)
                        rectangle = GT_syntext['wordBB'][0][i][:, :, kk]  # 2*4
                        X = np.int0(rectangle[:, [0, 2]])
                        coor.append(X)
                        #      image = cv2.imread(img_path)
                        #     text_image = image[X[1][0]:X[1][1], X[0][0]:X[0][1]]
                        #     all_text_image.append(text_image)  # as x in training dataset
                        tl = rectangle[:, 0]
                        tr = rectangle[:, 1]
                        br = rectangle[:, 2]
                        bl = rectangle[:, 3]
                        points = np.array([tl, tr, br, bl]).reshape((4, 1, 2))

                        # rect = cv2.minAreaRect(points)
                        height = int(max(abs(bl[1] - tl[1]), abs(br[1] - tr[1])))
                        width = int(max(abs(tr[0] - tl[0]), abs(br[0] - bl[0])))
                        if height > 400 or width > 3000:
                            continue
                        src_pts = points.astype("float32")
                        dst_pts = np.array([[0, 0],
                                            [width - 1, 0],
                                            [width - 1, height - 1],
                                            [0, height - 1]], dtype="float32")
                        # if im.shape[0] > 8000 or im.shape[1] > 8000: (but i dont read image in this for loop)
                        # continue

                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        MM.append(M)
                        H.append(height)
                        W.append(width)
                    #            image = cv2.imread(img_path)
                    #            warped = cv2.warpPerspective(image, M, (width, height))

                    for i in range(num):
                        A = []
                        A.append(img_path)
                        A.append(tex[i])
                        A.append(coor[i])
                        A.append(MM[i])
                        A.append(H[i])
                        A.append(W[i])
                        info_.append(A)
                self.info_=info_
            return info_
        # info_
        def gen_train(self):
            for i in range(len(self.info_)):  # range(len(self.GetData())?!
                idx = np.random.randint(0, len(self.info_))
                image = cv2.imread(self.info_[idx][0])  #: image path
                # if image.shape[0] > 8000 or image.shape[1] > 8000:
                #   continue
                M = self.info_[idx][3]
                H = self.info_[idx][4]
                W = self.info_[idx][5]
                warped = cv2.warpPerspective(image, M, (W, H))

                text_string_GT = self.info_[idx][1]  #: text
                X = self.info_[idx][2]  #: BBcoord
                text_image = warped  # [X[1][0]:X[1][1], X[0][0]:X[0][1]]
                yield (text_image, text_string_GT)

        def create_ds(self):
            image_and_label_ds = list()
            vocab = []
            max_length = -1
            max_width = -1

            for im in range(self.num_images_exist):  # num_total_text_images):
                filePath = os.path.join(self.out_path, str(im) + ".jpg")
                text_image, text_string_GT = next(self.gen_train())
                text_string_GT
                if len(text_string_GT) > max_length:
                    max_length = len(text_string_GT)
                # first resize  width based on h
                # then find max size
                # but !?wait! first batch and then do these
                if text_image.shape[1] > max_width:
                    pass
                cv2.imwrite(filePath, text_image)
                imglabel = [filePath, text_string_GT]
                image_and_label_ds.append(imglabel)
            return image_and_label_ds, max_length

        def gen_train(self):
            for i in range(len(self.info_)):
                idx = np.random.randint(0, len(self.info_))
                image = cv2.imread(self.info_[idx][0])  #: image path
                M = self.info_[idx][3]
                H = self.info_[idx][4]
                W = self.info_[idx][5]
                warped = cv2.warpPerspective(image, M, (W, H))

                text_string_GT = self.info_[idx][1]  #: text
                X = self.info_[idx][2]  #: BBcoord
                text_image = warped  # [X[1][0]:X[1][1], X[0][0]:X[0][1]]
                yield (text_image, text_string_GT)

    # %%

syntext_ds = SynthTextData()
ds_syntext = syntext_ds.create_ds()

text_image , text_string_GT = next(syntext_ds.gen_train())
print(text_image.shape)
plt.imshow(text_image)
plt.show()
print(text_string_GT)

'''import numpy as np
import os, traceback, datetime
from tqdm import tqdm
import cv2, re
from DataUtils.Utils import DataObject, inputF
import scipy.io

class SynthTextData():
    def __init__(self, datasetName, debug):
        self.rootImagesPath = "/media/Archive4TB3/Data/textImages/SynthText/"

        self.gtPath = os.path.join(self.rootImagesPath, "gt.mat")
        self.datasetName = datasetName
        self.out_path = "/media/Archive4TB3/Data/textImages/SynthText/SynthTextCroppedImages"
        self.testPercent = 0.1
        self.valPercent = 0.05
        self.max_width = -1
        self.max_height = -1
        self.debug = debug

    def getData(self):
        print("********** train Data **********")
        allData = self.getWord()
        np.random.shuffle(allData)
        idx = int((1.0 - self.testPercent - self.valPercent) * len(allData))
        trainData = allData[:idx]
        print("********** test Data **********")
        testData = allData[idx:idx + int(self.testPercent * len(allData))]
        idx += int(self.testPercent * len(allData))
        print("********** validation Data **********")
        valData = allData[idx:idx + int(self.valPercent * len(allData))]
        return trainData, testData, valData

    def getLine(self):
        if os.path.exists(self.out_path):
            answer = inputF("Do you want to remove {} DataSet?(y|n): ".format(self.out_path))
            if answer == "y":
                os.system("rm -rf {}".format(self.out_path))
            else:
                print("you should delete it, so first check it and try again")
                exit(1)
        os.makedirs(self.out_path, exist_ok=True)

        gt_mat = scipy.io.loadmat(self.gtPath)
        allData = []
        self.max_width = -1
        self.max_height = -1
        count = 0
        for imc in tqdm(range(len(gt_mat["imnames"][0]))):
            try:
                img_path = os.path.join(self.rootImagesPath, gt_mat["imnames"][0][imc][0])
                im = cv2.imread(img_path)
                if im.shape[0] > 8000 or im.shape[1] > 8000:
                    continue
                if self.debug:
                    print(gt_mat["txt"][0][imc])
                    cv2.imshow("", im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                lines = "\n".join(gt_mat["txt"][0][imc].tolist()).splitlines()
                wordc = 0
                for line in lines:
                    text = re.sub(' +', ' ', line.strip())
                    noWord = len(text.split(" "))
                    if len(lines) == 1:
                        wordsBB = gt_mat["wordBB"][0][imc][:, :]
                        wordsBB = np.expand_dims(wordsBB, axis=-1)
                    else:
                        wordsBB = gt_mat["wordBB"][0][imc][:, :, wordc:wordc+noWord]
                    wordc += noWord
                    tl = wordsBB[:, 0, np.argmin(wordsBB[0, 0, :])]
                    tr = wordsBB[:, 1, np.argmax(wordsBB[0, 1, :])]
                    br = wordsBB[:, 2, np.argmax(wordsBB[0, 2, :])]
                    bl = wordsBB[:, 3, np.argmin(wordsBB[0, 3, :])]

                    points = np.array([tl, tr, br, bl]).reshape((4, 1, 2))

                    # rect = cv2.minAreaRect(points)
                    height = int(max(abs(bl[1] - tl[1]), abs(br[1] - tr[1])))
                    width = int(max(abs(tr[0] - tl[0]), abs(br[0] - bl[0])))
                    if height > 400 or width > 3000:
                        continue
                    src_pts = points.astype("float32")
                    dst_pts = np.array([[0, 0],
                                        [width - 1, 0],
                                        [width - 1, height - 1],
                                        [0, height - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(im, M, (width, height))
                    if self.debug:
                        print(text)
                        cv2.imshow("", warped)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    if width > self.max_width:
                        self.max_width = width
                    if height > self.max_height:
                        self.max_height = height

                    chars = list(text)
                    filePath = os.path.join(self.out_path, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") +
                                                                            str(count) + ".jpg")
                    count += 1
                    cv2.imwrite(filePath, warped)
                    allData.append(DataObject(chars, filePath))

            except:
                traceback.print_exc()
                continue
        return allData

    def getWord(self):
        if os.path.exists(self.out_path):
            answer = inputF("Do you want to remove {} DataSet?(y|n): ".format(self.out_path))
            if answer == "y":
                os.system("rm -rf {}".format(self.out_path))
            else:
                print("you should delete it, so first check it and try again")
                exit(1)
        os.makedirs(self.out_path, exist_ok=True)

        gt_mat = scipy.io.loadmat(self.gtPath)
        allData = []
        self.max_width = -1
        self.max_height = -1
        count = 0
        for imc in tqdm(range(len(gt_mat["imnames"][0]))):
            try:
                img_path = os.path.join(self.rootImagesPath, gt_mat["imnames"][0][imc][0])
                im = cv2.imread(img_path)
                if im.shape[0] > 8000 or im.shape[1] > 8000:
                    continue
                if self.debug:
                    print(gt_mat["txt"][0][imc])
                    cv2.imshow("", im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                words = re.sub(' +', ' ', " ".join(gt_mat["txt"][0][imc].tolist()).replace("\n", " ").strip()).split(" ")
                for i in range(len(words)):
                    text = words[i]
                    wordBB = gt_mat["wordBB"][0][imc][:, :, i]
                    tl = wordBB[:, 0]
                    tr = wordBB[:, 1]
                    br = wordBB[:, 2]
                    bl = wordBB[:, 3]

                    points = np.array([tl, tr, br, bl]).reshape((4, 1, 2))

                    # rect = cv2.minAreaRect(points)
                    height = int(max(abs(bl[1] - tl[1]), abs(br[1] - tr[1])))
                    width = int(max(abs(tr[0] - tl[0]), abs(br[0] - bl[0])))
                    if height > 400 or width > 3000:
                        continue
                    src_pts = points.astype("float32")
                    dst_pts = np.array([[0, 0],
                                        [width - 1, 0],
                                        [width - 1, height - 1],
                                        [0, height - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(im, M, (width, height))
                    if self.debug:
                        print(text)
                        cv2.imshow("", warped)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    if width > self.max_width:
                        self.max_width = width
                    if height > self.max_height:
                        self.max_height = height

                    chars = list(text)
                    filePath = os.path.join(self.out_path, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") +
                                                                            str(count) + ".jpg")
                    count += 1
                    cv2.imwrite(filePath, warped)
                    allData.append(DataObject(chars, filePath))

            except:
                traceback.print_exc()
                continue
        return allData'''
