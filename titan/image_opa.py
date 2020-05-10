import numpy as np
import cv2


img = cv2.imread('E:/test/input/dataset (1).png', cv2.IMREAD_COLOR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

pix = np.array(img)[0][0]

for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        item = np.array(img)[i][j]
        if item[0]==pix[0] and item[1]==pix[1] and item[2]==pix[2]:
            img[i][j] = [255, 255, 255, 0]
        else:
            continue

cv2.imwrite('C:/Users/LHZ/Desktop/aa.png', img)