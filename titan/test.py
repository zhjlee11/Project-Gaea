import sys
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import datetime
import socket
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


BATCH_SIZE = 20
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image, c=3):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, c])

  return cropped_image

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image, c=3):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image, c=c)

  # random mirroring
  #image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


def preprocess_image_test(image, label):
  image = normalize(image)
  return image

def preprocess_image_test_nl(image):
  image = normalize(image)
  return image

def preprocess_image_train_nl(image, c=3):
  image = random_jitter(image, c=c)
  image = normalize(image)
  return image
  
def load_model():
    return tf.keras.models.load_model(os.path.join(os.getcwd(), 'saved_model', 'generator4.h5'))

    
def similar(n1, n2, d=0):
    if d == -1:
        return False
    sn = abs(n1-n2)
    if sn >= 0 and sn <= d :
        return True
    else :
        return False

class image_predict():
    def __init__(self, model, path, imn, degree=0):
        self.degree = degree
        self.model = model
        
        img = Image.open(path + "/" + imn).convert("RGBA")
        
        pix = np.array(img)[0][0]
        datas = img.getdata()
        newData = []
        for item in datas:
            if similar(item[0], pix[0], d=self.degree) and similar(item[1], pix[1], d=self.degree) and similar(item[2], pix[2], d=self.degree):
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        img = np.array(img)
        
        
        self.oimg = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        self.image = preprocess_image_train_nl(self.oimg, c=4)
        
        self.pred = None
        self.imn = imn
        
        

    def predict(self):
        pred = np.array(self.model(self.image[np.newaxis], training=True)[0]*0.5+0.5)
        self.pred = cv2.resize(pred * 255, dsize=(128, 192), interpolation=cv2.INTER_AREA)
        cv2.imwrite('aaaabbb.png', cv2.cvtColor(self.pred, cv2.COLOR_BGR2RGBA))
        
        
image_predict(load_model(), 'aa', 'inptt.png').predict()