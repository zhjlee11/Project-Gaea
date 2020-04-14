import sys
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
  
def load_model(path='saved_model/generator.h5'):
    return tf.keras.models.load_model(path)

    
def similar(n1, n2, d=60):
    sn = abs(n1-n2)
    if sn >= 0 and sn <= d :
        return True
    else :
        return False

class image_predict():
    def __init__(self, model, path, imn, degree=60, temppath='tempimage'):
        img = cv2.imread(path + "/" + imn)
        self.degree = degree
        self.model = model
        self.owidth = img.shape[1] * 4
        self.oheight = img.shape[0] * 4
        inp2 = cv2.vconcat([img, img, img, img])
        self.image = cv2.hconcat([inp2, inp2, inp2, inp2])
        self.oimg = cv2.resize(self.image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        self.image = normalize(self.oimg)
        self.pred = None
        self.imn = imn
        self.temppath = temppath

    def predict(self):
        pred = np.array(self.model(self.image[np.newaxis])[0])*0.5+0.5
        self.pred =  cv2.resize(pred * 255, dsize=(self.owidth, self.oheight), interpolation=cv2.INTER_AREA)
        
        #self.pred = (self.pred * 255 / np.max(self.pred*255)).astype('uint8')
        
        cv2.imwrite(self.temppath + "/"+self.imn, self.pred)
        
        img = Image.open(self.temppath + "/"+self.imn)
        try: 
            os.remove(self.temppath + "/"+self.imn)
        except:
            pass
        pix = np.array(img)[0][0]
        
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if similar(item[0], pix[0], d=self.degree) and similar(item[1], pix[1], d=self.degree) and similar(item[2], pix[2], d=self.degree):
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        self.pred = img
        return self.pred
        
def reset(path):
    for j in [file for file in os.listdir(path) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")]:
        try: 
            os.remove(path + "/" + j)
        except:
            pass
            
def savelog(path, index, lic="normal"):
    f = open(path + "/log.titan", 'w')
    f.write(
    '''Titan Program Log

convert index : {0}
license : {1}
convertday : {2}
'''.format(index, lic, datetime.datetime.today().strftime('%Y-%m-%d'))
    )
    f.close()
    
def deletelog(path):
    try:
        os.remove(path+'/log.titan')
    except:
        pass
    
def checklog(path):
    try :
        f = open(path + "/log.titan", 'r')
        num = f.readlines()[2].split("convert index : ")[-1].split("\n")[0]
        f.close()
        return num
    except:
        return -1

def convertimage(model, ipath, opath, degree, counter, series=-1):
    if series == -1 :
        ilist = sorted([file for file in os.listdir(ipath) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")])
        for i, file in enumerate(ilist):
            try :
                image_predict(model, ipath,  str(file), degree=degree).predict().save(opath + "/"+str(file)) 
            except:
                counter.maxindex -= 1
                pass
            savelog(opath, i)
            counter.convertpro.setValue(i+1)
            counter.convertpro.update()
            reset('tempimage')
        deletelog(opath)
        return
    else :
        
        ilist = sorted([file for file in os.listdir(ipath) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")])[int(series)+1:]
        for i, file in enumerate(ilist):
            try :
                image_predict(model, ipath,  str(file), degree=degree).predict().save(opath + "/"+str(file)) 
            except:
                counter.maxindex -= 1
                pass
            savelog(opath, i+int(series)+1)
            counter.convertpro.setValue(i+int(series)+1+1)
            counter.convertpro.update()
            reset('tempimage')
        deletelog(opath)
        return
        
def convert(model, ipath, opath, degree):
    num = checklog(opath)
    if num == -1:
        convertimage(model, ipath, opath, degree)
    else :
        convertimage(model, ipath, opath, degree, series=num)

class Titan(QMainWindow):
    def __init__(self):
        super().__init__()
        self.inputp = ""
        self.outputp = ""
        self.model = None
        self.index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Titan')
        self.setGeometry(300, 300, 800, 600)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        
        try :
            tfver = tf.__version__
            self.model = load_model()
        except :
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.critical(self, "경고", "이 기기는 타이탄 프로그램을 지원하지 않습니다.")
            sys.exit()
                  
        selectinput = QPushButton('&변환할 이미지가 있는 폴더를 선택하세요')
        selectinput.clicked.connect(self.selectInput)
        self.inputtext = QLabel('선택된 폴더가 없습니다.')
      
        selectoutput = QPushButton('&변환한 이미지를 저장할 폴더를 선택하세요')
        selectoutput.clicked.connect(self.selectOutput)
        self.outputtext = QLabel('선택된 폴더가 없습니다.')
        
        grid = QGridLayout()    
        grid.addWidget(selectinput, 0, 0)
        grid.addWidget(self.inputtext, 0, 1)
        grid.addWidget(selectoutput, 1, 0)
        grid.addWidget(self.outputtext, 1, 1)
        
        self.degreeslider = QSlider(Qt.Horizontal)
        self.degreeslider.setMaximum(255)
        self.degreeslider.setMinimum(0)
        self.degreeslider.setSingleStep(1)
        self.degreeslider.setPageStep(10)
        self.degreeslider.setValue(55)
        self.degreeslider.valueChanged.connect(self.changeslider)
        
        self.degreeslidertext = QLabel("투명화 감도 : " + str(self.degreeslider.value()))
        
        degreela = QHBoxLayout()
        degreela.addWidget(self.degreeslider)
        degreela.addWidget(self.degreeslidertext)
           
        convertb = QPushButton('&변환 시작')
        convertb.clicked.connect(self.convertstart)
        
        self.convertpro = QProgressBar()
        self.convertpro.setValue(0)
        self.convertpro.setMinimum(0)
        self.convertpro.setMaximum(1)
        
        
        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addLayout(degreela)
        layout.addWidget(convertb)
        layout.addWidget(self.convertpro)
 
        central_widget = QWidget()
        central_widget.setLayout(layout) 
        self.setCentralWidget(central_widget)
        
        self.show()
        
    def convertstart(self):
        model = self.model
        ipath = str(self.inputp)
        opath = str(self.outputp)
        degree = int(self.degreeslider.value())
        
        self.maxindex = len([file for file in os.listdir(ipath) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")])
        self.convertpro.setMaximum(self.maxindex)
        
        num = checklog(opath)
        if num == -1:
            convertimage(model, ipath, opath, degree, self)
            filen = self.convertpro
        else :
            filen = self.convertpro
            res = QMessageBox().question(self, '이어하기', '과거에 변환을 하던 기록이 있습니다. 이어서 변환할까요? (아니요를 누를 시 처음부터 다시 변환됩니다. 과거 변환 후에 파일이 추가되었다면 아니요를 눌러주세요.)', QMessageBox.Yes | QMessageBox.No)
            if res == QMessageBox.Yes :
                self.convertpro.setValue(int(num)+1)
                filen = self.maxindex-int(num)-1
                convertimage(model, ipath, opath, degree, self, series=num)
            elif res == QMessageBox.No :
                convertimage(model, ipath, opath, degree, self)
            else :
                convertimage(model, ipath, opath, degree, self)
                
        ret = QMessageBox()
        ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        ret.information(self, "정보", "{0}개의 이미지 변환 완료".format(filen))
        
                
    def changeslider(self):
        self.degreeslidertext.setText("투명화 감도 : " + str(self.degreeslider.value()))
        
    def selectInput(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.inputp = QFileDialog. getExistingDirectory(self,"변환할 이미지가 있는 폴더를 선택하세요." )
        self.inputtext.setText(str(self.inputp))
        
    def selectOutput(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.outputp = QFileDialog. getExistingDirectory(self,"변환한 이미지를 저장할 폴더를 선택하세요." )
        self.outputtext.setText(str(self.outputp))
 
def run():
    global index
    app = QApplication(sys.argv)
    ex = Titan()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
    exit()