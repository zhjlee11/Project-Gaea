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
import tflite_runtime.interpreter as tflite

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image
  
def load_model():
    return tf.lite.Interpreter(model_path="converted_model.tflite")
    
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
        self.image = normalize(self.oimg)
        
        self.pred = None
        self.imn = imn
        
        

    def predict(self):
        self.model.allocate_tensors()
        self.model.set_tensor(self.model.get_input_details()[0]['index'], np.array(self.image[np.newaxis], dtype=np.float32))
        self.model.invoke()
        pred = self.model.get_tensor(self.model.get_output_details()[0]['index'])
        pred = np.array((pred[0]*0.5+0.5) * 255)
        self.pred = cv2.resize(pred, dsize=(128, 192), interpolation=cv2.INTER_AREA)
        cv2.imwrite('aa/aaaabbb.png', cv2.cvtColor(self.pred, cv2.COLOR_BGR2RGBA))
        
        
image_predict(load_model(), 'aa', 'inptt.png').predict()