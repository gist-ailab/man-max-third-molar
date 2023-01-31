import os
import cv2
import glob


pano_image_train = glob.glob("./dataset/PNG_Images/train/*.png")
pano_image_val = glob.glob("./dataset/PNG_Images/val/*.png")
pano_image_test = glob.glob("./dataset/PNG_Images/test/*.png")

for img_tr in pano_image_train:
    img = cv2.imread("{}".format(img_tr))
    print(img.shape)
    