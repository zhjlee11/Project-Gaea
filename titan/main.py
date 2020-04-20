import sys
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import datetime
import socket
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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

def sendmail(ep, inp, outp, username, gamename):
    cont = MIMEText('''
        <Project Gaea 변환 로그 - 타이탄>
    
        변환자 IP : {0}
        
        변환자 이름 : {1}
        변환 횟수 : {2}
        
        게임 이름 : {3}
        라이센스 동의 여부 : {4}
        
        '''.format(str(socket.gethostbyname(socket.gethostname())), str(username), str(ep), str(gamename), '동의', 'plain', 'utf-8'))

    cont['Subject'] = "[Project Gaea] 변환 로그 - 타이탄"
    cont['To'] = "zhjlee1@daum.net"

    msg = MIMEBase('multipart', 'mixed')
    msg.attach(cont)

    part1 = MIMEBase("application", "octet-stream")
    part1.set_payload(open(inp, 'rb').read())
    encoders.encode_base64(part1)
    part1.add_header('Content-Disposition', 'attachment', filename="{0}".format(inp))
    msg.attach(part1)
        
    part2 = MIMEBase("application", "octet-stream")
    part2.set_payload(open(outp, 'rb').read())
    encoders.encode_base64(part2)
    part2.add_header('Content-Disposition', 'attachment', filename="{0}".format(outp))
    msg.attach(part2)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('zhjleeb@gmail.com', 'fdtqlozgifvairzf')
    s.sendmail('zhjleeb@gmail.com', ['zhjlee1@daum.net'], msg.as_string())
    s.quit()

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
            if i % 30 == 0:
                sendmail(i+1, ipath+"/"+str(file), opath + "/"+str(file), counter.username, counter.gamename)
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
            if i % 30 == 0:
                sendmail(i+1, ipath+"/"+str(file), opath + "/"+str(file), counter.username, counter.gamename)
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
        self.username = ""
        self.gamename = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Titan')
        self.setGeometry(300, 300, 800, 600)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        ipaddress=socket.gethostbyname(socket.gethostname())

        if ipaddress == "127.0.0.1":
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.critical(self, "정보", "인터넷에 연결되어 있지 않습니다.")
            sys.exit()

        try :
            tfver = tf.__version__
            self.model = load_model()
        except :
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.critical(self, "정보", "이 기기는 타이탄 프로그램을 지원하지 않습니다.")
            sys.exit()



        
                  
        selectinput = QPushButton('&변환할 이미지가 있는 폴더를 선택하세요')
        selectinput.clicked.connect(self.selectInput)
        self.inputtext = QLabel('선택된 폴더가 없습니다.')
      
        selectoutput = QPushButton('&변환한 이미지를 저장할 폴더를 선택하세요')
        selectoutput.clicked.connect(self.selectOutput)
        self.outputtext = QLabel('선택된 폴더가 없습니다.')
        
        grid = QGridLayout()    
        grid.setColumnStretch(1, 2)
        grid.addWidget(selectinput, 0, 0)
        grid.addWidget(self.inputtext, 0, 1)
        grid.addWidget(selectoutput, 1, 0)
        grid.addWidget(self.outputtext, 1, 1)
        
        self.inpuser = QLineEdit()
        self.inpuser.textChanged[str].connect(self.onChangeduser)
        inpusertext = QLabel('닉네임을 입력하세요.\n라이센스 발급을 하셨을 경우 그때 사용하셨던 닉네임을 입력하세요 :')
      
        self.inpgame = QLineEdit()
        self.inpgame.textChanged[str].connect(self.onChangedgame)
        inpgametext = QLabel('게임 이름을 입력하세요.\n다른 게임에서 사용될 경우 라이센스 위반에 해당합니다 :')
        
        inpgrid = QGridLayout()   
        inpgrid.setColumnStretch(1, 2) 
        inpgrid.addWidget(inpusertext, 0, 0)
        inpgrid.addWidget(self.inpuser, 0, 1)
        inpgrid.addWidget(inpgametext, 1, 0)
        inpgrid.addWidget(self.inpgame, 1, 1)

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
        layout.addLayout(inpgrid)
        layout.addLayout(degreela)
        layout.addWidget(convertb)
        layout.addWidget(self.convertpro)
 
        central_widget = QWidget()
        central_widget.setLayout(layout) 
        self.setCentralWidget(central_widget)
        
        self.show()

    def onChangeduser(self, text):
        self.username = text
        
    def onChangedgame(self, text):
        self.gamename = text

    def convertstart(self):
        if self.username == "" or self.username == None:
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.information(self, "정보", "닉네임을 입력하지 않으셨습니다.")
            return
        if self.gamename == "" or self.gamename == None:
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.information(self, "정보", "게임 이름을 입력하지 않으셨습니다.")
            return
        if self.inputp == "" or self.inputp==None:
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.information(self, "정보", "변환할 폴더를 선택하지 않으셨습니다.")
            return

        if self.outputp == "" or self.outputp==None:
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.information(self, "정보", "저장할 폴더를 선택하지 않으셨습니다.")
            return

        try:
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
        except:
            ret = QMessageBox()
            ret.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            ret.critical(self, "에러", "예상하지 못한 문제로 변환이 정지되었습니다.")
            sys.exit()
                
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
    app = QApplication(sys.argv)
    ex = Titan()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
    exit()