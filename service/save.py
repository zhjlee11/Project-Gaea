import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import datetime



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

def convert(model, ipath, opath, degree, series=-1):
    if series == -1 :
        ilist = sorted([file for file in os.listdir(ipath) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")])
        for i, file in enumerate(ilist):
            image_predict(model, ipath,  str(file), degree=degree).predict().save(opath + "/"+str(file)) 
            savelog(opath, i)
            reset('tempimage')
        deletelog(opath)
        return
    else :
        ilist = sorted([file for file in os.listdir(ipath) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")])[int(series)+1:]
        for i, file in enumerate(ilist):
            image_predict(model, ipath,  str(file), degree=degree).predict().save(opath + "/"+str(file)) 
            savelog(opath, i)
            reset('tempimage')
        deletelog(opath)
        return
        
def convertI(model, ipath, opath, degree):
    num = checklog(opath)
    if num == -1:
        convert(model, ipath, opath, degree)
    else :
        convert(model, ipath, opath, degree, series=num)

    

 
model = load_model()
ipath = 'F:/KaliNode/programming/Project-Gaea/Project-Gaea/service'
opath = 'F:/KaliNode/programming/Project-Gaea/Project-Gaea/service/converteddata'
convertI(model, ipath, opath, 60)


#image_predict(load_model(), '',  "inputtest.png").predict().save("./converteddata/transparent.png", "PNG") 