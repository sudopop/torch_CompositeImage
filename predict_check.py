import cv2
import numpy as np

#img = cv2.imread("/home/son/Work/Pytorch-UNet/sci_data/test_data/0.png")
img = cv2.imread("./2.jpg")
# img = cv2.imread("./3.jpg")
mask = cv2.imread("./output_2.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, normal_gray = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
# normal_gray = normal_gray.astype('int')
ano_img = cv2.bitwise_and(img, img, mask=normal_gray)

mask22 = cv2.cvtColor(ano_img, cv2.COLOR_BGR2GRAY)
_, aa = cv2.threshold(mask22, 240, 255, cv2.THRESH_BINARY)
ano_img2 = cv2.bitwise_and(img, img, mask=aa)
cc = ano_img-ano_img2

# cv2.imshow('aa', aa)
# cv2.imshow('ano_img', ano_img)
# cv2.imshow('3', aa)
# cv2.imshow('ano_img2', ano_img2)
# cv2.imshow('cc', cc)
# cv2.imshow('normal_gray', ano_img2)
# cv2.waitKey(0)


gray_result = cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w>20 and h >20:
        cv2.drawContours(cc, [cnt], 0, (0, 0, 255), 3)  # 이미지는 작게
        cc = cv2.rectangle(cc, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('result', cc)
# cv2.imshow('mask', mask)
# cv2.imshow('normal_gray', normal_gray)
cv2.waitKey(0)