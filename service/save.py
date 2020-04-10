import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
  
def load_model(path='saved_model/generator.h5'):
    return tf.keras.models.load_model(path)

#이미지를 Generator을 통해서 "여름->겨울"로 변환하는 함수
def generate_images(test_input, predicted):
    plt.figure(figsize=(12, 12))

    display_list = [test_input, predicted]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    
def similar(n1, n2, d=60):
    sn = abs(n1-n2)
    if sn >= 0 and sn <= d :
        return True
    else :
        return False
  
class image_predict():
    def __init__(self, model, img):
        self.model = model
        self.owidth = img.shape[1] * 4
        self.oheight = img.shape[0] * 4
        inp2 = cv2.vconcat([img, img, img, img])
        self.image = cv2.hconcat([inp2, inp2, inp2, inp2])
        self.oimg = cv2.resize(self.image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        self.image = normalize(self.oimg)
        self.pred = None

    def predict(self):
        pred = self.model(self.image[np.newaxis])
        self.pred = cv2.resize(np.float32(pred[0]) * 0.5 + 0.5, dsize=(self.owidth, self.oheight), interpolation=cv2.INTER_AREA)
        return self.pred
    
    def plt(self) :
        generate_images(self.image,self.model(self.image[np.newaxis])[0])
        
    def save(self):
        #if self.pred.all() == None:
            #self.predict()
        predi = np.array(self.model(self.image[np.newaxis])[0])*0.5+0.5
        cv2.imwrite('./converteddata/input.png', self.oimg)
        cv2.imwrite('./converteddata/pred.png', 255*predi)
        print(predi.shape)
        print("Save 완료")

	

'''
img = cv2.imread("inputtest.png")
#rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgp = image_predict(load_model(), img)
#imgp.plt()
#predicted_image = cv2.convertScaleAbs(imgp.predict(), alpha=(255.0))
#predicted_image = imgp.predict()
imgp.save()

'''


from PIL import Image
img = Image.open('./converteddata/pred.png')

pix = np.array(img)[0][0]

img = img.convert("RGBA")
datas = img.getdata()

newData = []
 
for item in datas:
    if similar(item[0], pix[0]) and similar(item[1], pix[1]) and similar(item[2], pix[2]):
        newData.append((255, 255, 255, 0))
        # RGB의 각 요소가 모두 cutOff 이상이면 transparent하게 바꿔줍니다.
    else:
        newData.append(item)
        # 나머지 요소는 변경하지 않습니다.
 
img.putdata(newData)
img.save("./converteddata/transparent.png", "PNG") # PNG 포맷으로 저장합니다.