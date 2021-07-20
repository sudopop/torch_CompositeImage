import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
import datetime
from random import randrange, randint

class Composite():
    def __init__(self, TrueImageDir="", img_save_path="", label_save_path=""):

        self.TrueImageDir = TrueImageDir
        self.Img_save_path = img_save_path
        self.Img_label_path = label_save_path

        self.Truedata = []  # List of image files for reader
        self.Truedata += [each for each in os.listdir(self.TrueImageDir) if
                          each.endswith('.PNG') or each.endswith('.JPEG') or each.endswith('.TIF') or each.endswith(
                              '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                              '.tif') or each.endswith('.gif')]  # Get list of training images
        self.Truedata.sort()


    def getitem(self, idx, bg_true=False):
        # 정상제품이 아래(bg) 있으면 True, 위(img)에 있으면 False
        # TrueName = self.Truedata[idx]
        tmp = randrange(len(self.Truedata)) #이물 이미지가 절대적으로 많으므로
        TrueName = self.Truedata[tmp]
        true_name = join(self.TrueImageDir, TrueName)

        bg_img = cv2.imread(true_name)  # 대파
        height, width, channel = bg_img.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 270, 1)
        bg_img = cv2.warpAffine(bg_img, matrix, (width, height))

        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        min_contour_color = 250
        max_contour_color = 255
        _, inv_bg_mask = cv2.threshold(bg_gray, min_contour_color, max_contour_color, cv2.THRESH_BINARY_INV)

        composite_label_np = np.array(inv_bg_mask)
        composite_label_np = composite_label_np / 255  # (***라벨링 시각화****) 레이블링하게 255 -> 1로
        composite_label_np = composite_label_np.astype(np.uint8)
        composite_label_np = Image.fromarray(composite_label_np)

        basename = "image"
        suffix = datetime.datetime.utcnow().strftime("%y%m%d_%H%M%S_%f_270")
        date_filename = "_".join([basename, suffix])

        img_filename = date_filename + ".jpg"
        img_filename = join(self.Img_save_path, img_filename)
        cv2.imwrite(img_filename, bg_img)

        labe_filename = date_filename + "_mask.png"
        labe_filename = join(self.Img_label_path, labe_filename)
        composite_label_np.save(labe_filename)

        # cv2.imshow('composite_img', bg_img)
        # cv2.imshow('composite_img2', inv_bg_mask)
        # cv2.waitKey(0)

def main():

    True_path = "./Only_Normal_400/"
    img_save_path = "./Only_Normal_400_270/"
    label_save_path = "./Only_Normal_400_label_270/"

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    data = Composite(True_path, img_save_path, label_save_path)

    for idx in range(len(data.Truedata)):
        print(idx)
        data.getitem(idx, True) # True: 정상제품이 아래 / False: 정상제품이 위

main()