import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
import datetime
from random import randrange, randint

class Composite():
    def __init__(self, AnomalyImageDir="", TrueImageDir="", img_save_path="", label_save_path="", rgb_version = True):
        self.AnomalyImageDir = AnomalyImageDir
        self.TrueImageDir = TrueImageDir
        self.Img_save_path = img_save_path
        self.Img_label_path = label_save_path
        self.Anormalydata = []  # List of image files for reader
        self.Anormalydata += [each for each in os.listdir(self.AnomalyImageDir) if
                              each.endswith('.PNG') or each.endswith('.JPEG') or each.endswith('.TIF') or each.endswith(
                                  '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                                  '.tif') or each.endswith('.gif')]  # Get list of training images
        self.Truedata = []  # List of image files for reader
        self.Truedata += [each for each in os.listdir(self.TrueImageDir) if
                          each.endswith('.PNG') or each.endswith('.JPEG') or each.endswith('.TIF') or each.endswith(
                              '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                              '.tif') or each.endswith('.gif')]  # Get list of training images
        self.Truedata.sort()

        self.rgb_version = rgb_version

    def getitem(self, idx, bg_true=False):
        # 정상제품이 아래(bg) 있으면 True, 위(img)에 있으면 False
        TrueName = self.Truedata[idx]
        true_name = join(self.TrueImageDir, TrueName)
        AnomalyName = self.Anormalydata[idx]
        anomaly_name = join(self.AnomalyImageDir, AnomalyName)

        true_img = cv2.imread(true_name)
        bg_gray = cv2.cvtColor(true_img, cv2.COLOR_BGR2GRAY)
        bg_img = cv2.imread(anomaly_name)  # 배경
        print(anomaly_name)
        ano_img = cv2.resize(bg_img, dsize=(true_img.shape[1], true_img.shape[0]),
                            interpolation=cv2.INTER_AREA)  # 정상에 맞게 사이즈 조정

        # cv2.imshow('composite_img', bg_gray)
        # cv2.waitKey(0)

        min_contour_color = 250
        max_contour_color = 255

        _, normal_gray_img = cv2.threshold(bg_gray, min_contour_color, max_contour_color, cv2.THRESH_BINARY_INV)  # foreground만 검정 이미지
        _, normal_gray_label = cv2.threshold(bg_gray, min_contour_color, max_contour_color,
                                       cv2.THRESH_BINARY_INV)  # foreground만 검정 이미지
        contours, hierarchy = cv2.findContours(normal_gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(normal_gray_img, [cnt], 0, (0, 0, 0), 3)  # 이미지는 작게
            cv2.drawContours(normal_gray_label, [cnt], 0, (0, 0, 0), 0)  # 레이블은 크게 해야 학습할때 파보다 조금 더 크게 학습함

        _, ano_gray = cv2.threshold(normal_gray_img, min_contour_color, max_contour_color, cv2.THRESH_BINARY_INV)  # foreground만 검정 이미지

        true_img = cv2.bitwise_and(true_img, true_img, mask=normal_gray_img)
        ano_img = cv2.bitwise_and(ano_img, ano_img, mask=ano_gray)
        composite_img = cv2.add(true_img, ano_img)  # 합성 이미지 결 #RGB로 학습시 사용

        # cv2.imshow('ano_gray', true_img)
        # cv2.imshow('normal_gray_img', ano_img)
        # cv2.imshow('composite_img', normal_gray_img)
        # cv2.imshow('composite_imga', ano_gray)
        # cv2.waitKey(0)

        composite_label_np = np.array(normal_gray_label)
        composite_label_np = composite_label_np / 1  # (***라벨링 시각화****) 레이블링하게 255 -> 1로
        composite_label_np = composite_label_np.astype(np.uint8)
        composite_label_np = Image.fromarray(composite_label_np)

        basename = "image"
        suffix = datetime.datetime.utcnow().strftime("%y%m%d_%H%M%S_%f")
        date_filename = "_".join([basename, suffix])

        img_filename = date_filename + ".jpg"
        img_filename = join(self.Img_save_path, img_filename)
        # composite_img = cv2.GaussianBlur(composite_img, (5, 5), 0)
        # composite_img = cv2.GaussianBlur(composite_img, (5, 5), 0)
        # composite_img = cv2.GaussianBlur(composite_img, (5, 5), 0)
        cv2.imwrite(img_filename, composite_img)

        labe_filename = date_filename + "_mask.png"
        labe_filename = join(self.Img_label_path, labe_filename)
        composite_label_np.save(labe_filename)

def main():

    True_path = "/home/son/Work/Pytorch-UNet/create_dataset/food_dataset/compoiste"

    dir_name = []
    dir_name += [each for each in os.listdir("/home/son/Work/Pytorch-UNet/create_dataset/food_dataset/composite_sample/")]
    iid = 0
    for iidx in range(len(dir_name)):
        Anomaly_path = "/home/son/Work/Pytorch-UNet/create_dataset/food_dataset/composite_sample/"
        dataset_name = dir_name[iidx]
        Anomaly_path = Anomaly_path + dataset_name
        img_save_path = "/home/son/Work/Pytorch-UNet/create_dataset/food_dataset/result"
        # img_save_path = img_save_path + dataset_name + "_train"
        label_save_path = "/home/son/Work/Pytorch-UNet/create_dataset/food_dataset/result"
        # label_save_path = label_save_path + dataset_name + "_label"

        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)

        data = Composite(Anomaly_path, True_path, img_save_path, label_save_path, True)

        # for idx in range(len(data.Anormalydata)-1):

        for idx in range(5): #수를 정하면 데이터가 많아도 딱 그만큼만 가능함
            print(iid)
            data.getitem(iid, True) # True: 정상제품이 아래 / False: 정상제품이 위
            iid +=1

main()