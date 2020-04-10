from flask import g, Flask, render_template, request, Response, send_file, redirect, url_for, make_response, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from PIL import Image

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
  
def load_model(path='saved_model/generator.h5'):
    return tf.keras.models.load_model(path)
    
def get_ip(e):
  try:
    return e["HTTP_X_FORWARDED_FOR"].split(",")[0].strip()
  except (KeyError, IndexError):
    return e.get("REMOTE_ADDR")
    
def similar(n1, n2, d=60):
    sn = abs(n1-n2)
    if sn >= 0 and sn <= d :
        return True
    else :
        return False
  
    

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
  
class image_predict():
    def __init__(self, model, img):
        self.model = model
        self.owidth = img.shape[1] * 4
        self.oheight = img.shape[0] * 4
        inp2 = cv2.vconcat([img, img, img, img])
        self.image = cv2.hconcat([inp2, inp2, inp2, inp2])
        self.oimg = cv2.resize(self.image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        self.image = normalize(self.oimg)

        
    
    def predict(self):
        pred = np.array(self.model(self.image[np.newaxis])[0])*0.5+0.5
        return cv2.resize(pred * 255, dsize=(self.owidth, self.oheight), interpolation=cv2.INTER_AREA)
    
    def plt(self) :
        generate_images(self.image,self.model(self.image[np.newaxis])[0])


@app.route('/')
def default_template():
    try:
        path = session['saved_image']
        if path != None:
            os.remove(path)
    except:
        pass
    return render_template('main.html')
    
@app.route('/info')
def info_template():
    return render_template('info.html')
    
@app.route('/license')
def license_template():
    return render_template('license.html')

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
def upload_file(alert=None):
    try:
        path = session['saved_image']
        if path != None:
            os.remove(path)
    except:
        pass
    if request.method == 'POST':
        try:
            path = session['saved_image']
            if path != None:
                os.remove(path)
        except:
            pass
        
        f = request.files['input-file-preview']
        gamename = request.form['gamename']
        agree = request.form.get('agree')
        if agree != 'on' :
            alert = "라이센스에 동의하지 않으셨습니다."
            return render_template('upload.html', message=alert)
        if gamename == False or gamename == None or gamename == "" or gamename == " ":
            alert = "정상적인 게임 이름이 아닙니다."
            return render_template('upload.html', message=alert)
        if f == None :
            alert = "비어있는 파일"
            return render_template('upload.html', message=alert)
        try :
            f.save("./data/" + secure_filename(f.filename))
        except:
            alert = "비어있는 파일"
            return render_template('upload.html', message=alert)
        imga = cv2.imread("./data/" + secure_filename(f.filename))
        imgp = image_predict(get_model(), imga)
        path = './converteddata/'+secure_filename(f.filename)
        predicted_image = imgp.predict()
        cv2.imwrite(path, predicted_image)
        
        img = Image.open(path)

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
        img.save(path, "PNG") # PNG 포맷으로 저장합니다.
        
        session['saved_image'] = path
        
        if agree == 'on':
            agreemes = "동의"
        elif agree != 'on':
            agreemes = "비동의"
        else :
            alert = "예기치 못한 오류"
            return render_template('upload.html', message=alert)
        
        cont = MIMEText('''
        <Project Gaea 변환 로그>
        
        변환자 IP : {0}
        
        게임 이름 : {1}
        라이센스 동의 여부 : {2}
        
        '''.format(get_ip(request.environ), str(gamename), str(agreemes)), 'plain', 'utf-8')
        cont['Subject'] = "[Project Gaea] 변환 로그"
        cont['To'] = "zhjlee1@daum.net"

        msg = MIMEBase('multipart', 'mixed')
        msg.attach(cont)

        path1 = "./data/" + secure_filename(f.filename)
        part1 = MIMEBase("application", "octet-stream")
        part1.set_payload(open(path1, 'rb').read())
        encoders.encode_base64(part1)
        part1.add_header('Content-Disposition', 'attachment', filename="{0}".format(path1))
        msg.attach(part1)
        
        path2 = './converteddata/'+secure_filename(f.filename)
        part2 = MIMEBase("application", "octet-stream")
        part2.set_payload(open(path2, 'rb').read())
        encoders.encode_base64(part2)
        part2.add_header('Content-Disposition', 'attachment', filename="{0}".format(path2))
        msg.attach(part2)

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('zhjleeb@gmail.com', 'fdtqlozgifvairzf')
        s.sendmail('zhjleeb@gmail.com', ['zhjlee1@daum.net'], msg.as_string())
        s.quit()
        
        try: 
            os.remove("./data/" + secure_filename(f.filename))
        except:
            pass
        
        return send_file(path, attachment_filename=secure_filename(f.filename), as_attachment=True)
    else:
        alert = "메서드 오류"
        return render_template('upload.html', message=alert)
        



app.run(debug = True)