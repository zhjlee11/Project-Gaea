from flask import g, Flask, render_template, request, Response, send_file, redirect, url_for, make_response, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

app = Flask(__name__)
app.secret_key = 'any random string'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

def get_model():
  gan = getattr(g, '_gan', None)
  if gan is None:
    gan = g._gan = load_model()
  return gan

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
  
def load_model(path='saved_model/generator_g'):
    return tf.keras.models.load_model(path)
    

#이미지를 Generator을 통해서 "여름->겨울"로 변환하는 함수
def generate_images(test_input, predicted):
    plt.figure(figsize=(12, 12))

    display_list = [test_input, predicted]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
  
class image_predict():
    def __init__(self, model, img):
        self.model = model
        self.owidth = img.shape[1] * 4
        self.oheight = img.shape[0] * 4
        self.oimg = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        self.image = normalize(self.oimg)
    
    def predict(self):
        pred = self.model(self.image[np.newaxis])
        return cv2.resize(np.float32(pred[0]), dsize=(self.owidth, self.oheight), interpolation=cv2.INTER_AREA)

@app.route('/')
def default_template():
    return render_template('main.html')
    
@app.route('/info')
def info_template():
    return render_template('info.html')

@app.route('/upload', methods = ['GET'])
def load_file():
    try:
        path = session['saved_image']
        if path != None:
            os.remove(path)
    except:
        pass
    return render_template('upload.html')
	
@app.route('/upload', methods = ['POST'])
def upload_file():
    try:
        path = session['saved_image']
        if path != None:
            os.remove(path)
    except:
        pass
    if request.method == 'POST':
        f = request.files['input-file-preview']
        if f == None :
            return "비어있는 파일"
        f.save("./data/" + secure_filename(f.filename))
        img = cv2.imread("./data/" + secure_filename(f.filename), 3)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgp = image_predict(get_model(), rgb_image)
        os.remove("./data/" + secure_filename(f.filename))
        path = './converteddata/'+secure_filename(f.filename)
        predicted_image = cv2.convertScaleAbs(imgp.predict(), alpha=(255.0))
        cv2.imwrite(path, predicted_image)
        session['saved_image'] = path
        return send_file(path, attachment_filename=secure_filename(f.filename), as_attachment=True)
    else:
        return "메서드 오류"
        



app.run(debug = True)