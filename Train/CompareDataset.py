import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import shutil
import os

class compareImg : 
    def __init__(self) : 
        pass 
        
    def readImg(self, filepath) : 
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
        return img 
        
    def diffImg(self, img1, img2) : 
        # Initiate SIFT detector 
        orb = cv2.ORB_create() # find the keypoints and descriptors with SIFT 
        
        kp1, des1 = orb.detectAndCompute(img1, None) 
        kp2, des2 = orb.detectAndCompute(img2, None) # create BFMatcher object 
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Match descriptors. 
        matches = bf.match(des1,des2) # Sort them in the order of their distance. 
        matches = sorted(matches, key = lambda x:x.distance) # BFMatcher with default params 
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2) # Apply ratio test 
        
        good = [] 
        for m,n in matches: 
            if m.distance < 0.75 * n.distance: 
                good.append([m]) # Draw first 10 matches.

        height1 = img1.shape[0]
        width1 = img1.shape[1]
        height2 = img2.shape[0] 
        width2 = img2.shape[1]

        pixels = height1 * width1 + height2 * width2
        pe = len(good)*2/pixels
        
        #knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2) 
        #print("유사도 : {0}%".format(pe * 100))
        return pe * 100
        #plt.imshow(knn_image) 
        #plt.show() 
        
    def run(self, filepath1, filepath2) : # 이미지 파일 경로 설정     
        img1 = self.readImg(filepath1) 
        img2 = self.readImg(filepath2) # 2개의 이미지 비교 
        return self.diffImg(img1, img2) 
        
if __name__ == '__main__': 
    cImg = compareImg()
    
    # 1.05 >= 유사도 > 0.025
    PATH_DIR='./Dataset/'
    CPATH_DIR='./RDataset/'

    for i in [file for file in os.listdir(PATH_DIR) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge")]:
        for j in [file for file in os.listdir(CPATH_DIR) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpge")]:
            perc = cImg.run(CPATH_DIR+str(j), PATH_DIR+str(i))
            if(perc > 0.025 and perc <= 1.05) :
                print("[적합]\n" + CPATH_DIR+str(j) + "와 "+PATH_DIR+str(i) + "의 유사도 : " + str(perc) + "%")
                shutil.copyfile(PATH_DIR+str(i), CPATH_DIR+str(i))
            else :
                print("[부적합]\n" + CPATH_DIR+str(j) + "와 "+PATH_DIR+str(i) + "의 유사도 : " + str(perc) + "%")
            print("-----------------------------------------------------------------------------------------------------------------------")
