#!/usr/bin/env python
# coding: utf-8

# # < CycleGAN - Summer To Winter >
# 
# 이 프로젝트는 CycleGAN을 통하여 여름 풍경 사진을 겨울 풍경 사진으로 변환하는 프로젝트입니다.

# In[ ]:


get_ipython().system('pip install --upgrade tensorflow')


# In[ ]:


get_ipython().system('pip install h5py pyyaml')


# ## (1) 데이터셋 다운로드

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals


import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from IPython.display import clear_output


from PIL import Image
import cv2


AUTOTUNE = tf.data.experimental.AUTOTUNE


# Tensorfow Dataset에서 summer2winter_yosemite 데이터 셋을 가져온 후, 분리한다.

# In[59]:


BUFFER_SIZE = 1000
BATCH_SIZE = 20
IMG_WIDTH = 256
IMG_HEIGHT = 256


# ## (2) 데이터셋 학습 전 준비 과정

# In[60]:


#이미지 주어진 크기에 따라 랜덤으로 분할,
def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image


# In[61]:


# "-1 <= image <= 1" 로 변환
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image


# In[ ]:





# In[62]:


#이미지 변환 처리 중..
def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image


# In[63]:


def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


# In[ ]:





# In[64]:


def preprocess_image_test(image, label):
  image = normalize(image)
  return image


# In[65]:


def preprocess_image_train_nl(image):
  image = random_jitter(image)
  image = normalize(image)
  return image


# In[66]:


class imagedata:
  def __init__(self, path,  w = 128, h = 192):
    self.outp = Image.open(path).convert("RGB")
    
    self.outp = self.outp.resize((w, h))
    area = (0, 0, int(w/4), int(h/4))
    self.inp = self.outp.crop(area)
    
    
    inp1 = cv2.cvtColor(np.array(self.inp), cv2.COLOR_BGR2RGB)
    inp2 = cv2.vconcat([inp1, inp1, inp1, inp1])
    self.inp = cv2.hconcat([inp2, inp2, inp2, inp2]) 
    
    
    self.oinp = self.inp
    self.ooutp = self.outp
    self.outp = np.array(self.ooutp)
    self.outp = cv2.cvtColor(self.outp, cv2.COLOR_BGR2RGB)
    self.inp = preprocess_image_train_nl(self.inp)
    self.outp = preprocess_image_train_nl(self.outp)


# In[67]:


'''PATH_DIR='./Dataset/'
filelist = []

for i in [file for file in os.listdir(PATH_DIR) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")]:
  try :
    filelist.append(imagedata(PATH_DIR + str(i)))
  except :
    print(str(i)+" 파일 로드 실패")
    continue
  print(str(i)+" 파일 로드 완료")'''


# In[68]:


PATH_DIR='./Dataset/'
filelist = [file for file in os.listdir(PATH_DIR) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")]


# ## (3) 각 데이터셋의 첫 데이터를 띄운다.

# ## (4) CycleGAN 신경망 구성
# > https://subinium.github.io/introduction-to-normalization/ : 정규화에 대한 글

# In[69]:


OUTPUT_CHANNELS = 3


# In[70]:


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# In[71]:


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# In[72]:


def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# In[73]:


LAMBDA = 100


# In[74]:


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# In[75]:


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# In[76]:


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[77]:


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


# ## (5) 신경망 준비
# *Generator* 함수 **G**와 **F**는 ***역함수*** 관계</br>
# *Discriminator* 함수 **X**와 **Y**는 ***역함수*** 관계

# In[78]:


generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# ## (9) 학습 준비

# In[79]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# In[80]:


#학습 횟수
EPOCHS = 30000


# In[81]:


#이미지를 Generator을 통해서 "여름->겨울"로 변환하는 함수
def generate_images(model, test_input):
  prediction = model(test_input[np.newaxis], training=True)
  test_input1 = cv2.resize(test_input[np.newaxis][0], dsize=(128, 192), interpolation=cv2.INTER_AREA)
  prediction1 = cv2.resize(np.array(prediction[0]), dsize=(128, 192), interpolation=cv2.INTER_AREA)
  plt.figure(figsize=(12, 12))


  display_list = [test_input1, prediction1]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  
  plt.show()


# In[82]:


#이미지를 Generator을 통해서 "여름->겨울"로 변환하는 함수
def generate_images_r(model, test_input, real_output):
  prediction = model(test_input[np.newaxis])
  test_input1 = cv2.resize(test_input[np.newaxis][0], dsize=(128, 192), interpolation=cv2.INTER_AREA)
  prediction1 = cv2.resize(np.array(prediction[0]), dsize=(128, 192), interpolation=cv2.INTER_AREA)
  plt.figure(figsize=(12, 12))


  real_output1 = cv2.resize(real_output[np.newaxis][0], dsize=(128, 192), interpolation=cv2.INTER_AREA)
  display_list = [test_input1, prediction1, real_output1]
  title = ['Input Image', 'Predicted Image', 'Real Image']
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')

  
  plt.show()


# In[83]:


def sampling(train_x, train_y, batch_size) :
  train_xb = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
  train_yb = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
  listaa = np.arange(train_x.shape[0])
  listaa = np.random.choice(listaa, batch_size, replace=False)
  for i in listaa:
    train_xb = np.append(train_xb, train_x[i][np.newaxis], axis=0)
    train_yb = np.append(train_yb, train_y[i][np.newaxis], axis=0)
  return train_xb, train_yb


# ## (10) 학습 함수 구성

# In[84]:


@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    
  return gen_total_loss


# In[85]:


@tf.function
def cal_g_loss(input_image, target, generator, discriminator):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
  return gen_total_loss


# In[86]:


'''train_x = []
train_y = []

for i in filelist:
  print("인풋 배열 크기 : " + str(i.inp.shape))
  print("아웃풋 배열 크기" + str(i.outp.shape))
  train_x.append(i.inp)
  train_y.append(i.outp)'''


# In[87]:


from random import sample
import random
def sampling_list(a, b, batch):
  batches = [random.randint(0,len(a)-1) for r in range(0, batch)]
  c = []
  d = []
  for j in batches:
    c.append(a[j])
    d.append(b[j])
  g = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
  h = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
  for i in range(0, batch):
    g = np.append(g, c[i][np.newaxis], axis=0)
    h = np.append(h, d[i][np.newaxis], axis=0)
  return g, h


# In[88]:


from random import sample
#filelist.append(imagedata(PATH_DIR + str(i)))
def sampling_batch(fl, b):
    #while True :
        #try :
            pd='./Dataset/'
            g = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
            h = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
            for i in sample(fl,b):
                g = np.append(g, imagedata(pd+str(i)).inp[np.newaxis], axis=0)
                h = np.append(h, imagedata(pd+str(i)).outp[np.newaxis], axis=0)
            return g, h
        #except :
            #continue


# In[89]:


'''train_x = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')
train_y = np.empty(shape=[0, IMG_HEIGHT, IMG_WIDTH, 3], dtype='float32')

numi = 1
for i in filelist:
  #print("인풋 배열 크기 : " + str(i.inp.shape))
  #print("아웃풋 배열 크기" + str(i.outp.shape))
  #train_x.append(i.inp)
  print(str(numi) + "번째 이미지 로드 중..")
  
  train_x = np.append(train_x, i.inp[np.newaxis], axis=0)
  train_y = np.append(train_y, i.outp[np.newaxis], axis=0)
  
  if train_x.shape[0] % 500 == 0 and train_y.shape[0] % 500 == 0:
    print("================================================================================================================")
    np.save("saved_array/train_x", train_x)
    np.save("saved_array/train_y", train_y)
    f = open("saved_array/textnum.txt", 'w')
    f.write(str(numi))
    f.close()
    
    print("= "+str(numi)+"번쨰 배열 파일 세이브 완료")
    print("================================================================================================================")
  numi = numi + 1
  #train_y.append(i.outp)
    '''


# In[90]:


def readago():
    f = open("saved_loss/textnum.txt", "r")
    lines = f.read()
    f.close()
    losstrainlist = []
    losstestlist = []
    for i in lines.split("\n"):
        if i == "[losslog]" :
            continue
        losstrainlist.append(float(i.split(":")[1]))
        losstestlist.append(float(i.split(":")[2]))
        
    return losstrainlist, losstestlist


# In[91]:


def drawlossg(ll1, ll2, epoch):
    y1 = ll1
    x1 = range(1, len(ll1)+1)
    y2 = ll2
    x2 = range(1, len(ll2)+1)
    
    plt.plot(x1, y1, x2, y2)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss Graph (Learning : {0})".format(epoch))
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()


# In[92]:


def saveloss(epoch, loss1, loss2):
    f = open("saved_loss/textnum.txt", "a")
    f.write("\n{0}:{1}:{2}".format(str(epoch), str(loss1.numpy()), str(loss2.numpy())))
    f.close()
    return loss1.numpy(), loss2.numpy()


# In[93]:


PATH_DIR='./testinput/'
testlist = []

for i in [file for file in os.listdir(PATH_DIR) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge") or file.endswith(".bmp")]:
  try :
    testlist.append(imagedata(PATH_DIR + str(i)))
  except :
    print(str(i)+" 파일 로드 실패")
    continue
  print(str(i)+" 파일 로드 완료")

test_x = []
test_y = []
for i in testlist:
  test_x.append(i.inp)
  test_y.append(i.outp)


# In[94]:



a, b = sampling_batch(filelist, BATCH_SIZE)
print("학습 인풋의 형태 {0}".format(a.shape))
print("학습 아웃풋의 형태 {0}".format(b.shape))
plt.figure(figsize=(12, 12))
display_list = [a[0], b[0]]
title = ['Input Image', 'Output Image']
for i in range(2):
  plt.subplot(1, 2, i+1)
  plt.title(title[i])
  plt.imshow(display_list[i] * 0.5 + 0.5)
  plt.axis('off')

plt.show()


# ## (11) 학습 시작

# In[ ]:


losstrainlist, losstestlist = readago()
cepoch=len(losstrainlist)

print("학습 데이터 : {0}장".format(len(filelist)))
print("배치사이즈 : {0}장".format(BATCH_SIZE))
for epoch in range(EPOCHS):


  #a, b = sampling_list(train_x, train_y, BATCH_SIZE)
  print("{0}번째 데이터셋 샘플링 시작".format(epoch+cepoch+1))
  a, b = sampling_batch(filelist, BATCH_SIZE)
  c, d = sampling_list(test_x, test_y, 1)
  print("{0}번째 학습 시작".format(epoch+cepoch+1))
  for n in range(0, BATCH_SIZE):
      losstrain = train_step(a[n][np.newaxis], b[n][np.newaxis])
  losstest = cal_g_loss(c, d, generator, discriminator)
  print("{0}번째 학습 완료".format(epoch+cepoch+1))
  loss1, loss2 = saveloss(epoch+cepoch+1, losstrain, losstest)
  losstrainlist.append(loss1)
  losstestlist.append(loss2)
  print("{0}번째 학습 로그 작성 완료".format(epoch+cepoch+1))
  '''
  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_x, train_y)):
    print("for문 진입")
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n+=1'''

  
  
  

  clear_output(wait=True)
  # Using a consistent image (sample_summer) so that the progress of the model
  # is clearly visible.
  #print("=================================={0}번째 학습 이미지 비교================================".format(epoch+cepoch+1))
  #generate_images_r(generator, a[0], b[0])
  print("=================================={0}번째 테스트 이미지 비교================================".format(epoch+cepoch+1))
  generate_images_r(generator, c[0], d[0])
  print("=================================={0}번째 손실 그래프================================".format(epoch+cepoch+1))
  drawlossg(losstrainlist, losstestlist, epoch+cepoch+1)
  try :
    if loss2 - losstestlist[-2] >= 0:
      print("{0}번째 손실 : {1} [+{2}]".format(epoch+cepoch+1, loss2, loss2 - losstestlist[-2]))
    else :
      print("{0}번째 손실 : {1} [{2}]".format(epoch+cepoch+1, loss2, loss2 - losstestlist[-2]))
  except:
    print("{0}번째 손실 : {1}".format(epoch+cepoch+1, loss2))

  if (epoch+cepoch+1) % 10 == 0:
    try :
      checkpoint.save(file_prefix = checkpoint_prefix)
    except :
      pass


# ## (12) 테스트 데이터에서 몇개를 추출하여 변환 테스트

# In[ ]:


generate_images_r(generator_g, train_x[0], train_y[0])


# In[ ]:


generator_g.save('saved_model/generator_g', save_format='tf')
print("generator_g 저장 완료")
generator_f.save('saved_model/generator_f', save_format='tf')
print("generator_f 저장 완료")
discriminator_x.save('saved_model/discriminator_x', save_format='tf')
print("discriminator_x 저장 완료")
discriminator_y.save('saved_model/discriminator_y', save_format='tf')
print("discriminator_y 저장 완료")


# In[ ]:


model = tf.keras.models.load_model('saved_model/generator_g')
generate_images_r(model, train_x[0], train_y[0])


# In[54]:


generator.save('saved_model/generator.h5')
discriminator.save('saved_model/discriminator.h5')


# In[ ]:




