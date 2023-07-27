from UI import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from os.path import join
from os import listdir
import numpy as np
import cv2

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
      
    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.open_file_1)
        self.ui.pushButton_2.clicked.connect(self.open_file_2)
        self.ui.pushButton_3.clicked.connect(self.keypoints)
        self.ui.pushButton_4.clicked.connect(self.match_keypoints)
        
    def open_file_1(self):
        self.filename_1_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.filename_1_path = self.filename_1_path.split('/')[-3]+ '\\' + self.filename_1_path.split('/')[-2] + '\\' + self.filename_1_path.split('/')[-1]
        self.filename_1 = cv2.imread(self.filename_1_path)
        
    def open_file_2(self):
        self.filename_2_path, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.filename_2_path = self.filename_2_path.split('/')[-3]+ '\\' + self.filename_2_path.split('/')[-2] + '\\' + self.filename_2_path.split('/')[-1]
        self.filename_2 = cv2.imread(self.filename_2_path)
        
    def keypoints(self):
        img_gray_1 = cv2.cvtColor(self.filename_1,cv2.COLOR_BGR2GRAY)
        features = cv2.SIFT_create()
        keypoints = features.detect(img_gray_1,None)
        output_image = cv2.drawKeypoints(img_gray_1, keypoints, 0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('img',output_image)
        cv2.imwrite('keypoints.png',output_image)
        cv2.waitKey(0)
        
    def match_keypoints(self):
        img_gray_1 = cv2.cvtColor(self.filename_1,cv2.COLOR_BGR2GRAY)
        img_gray_2 = cv2.cvtColor(self.filename_2,cv2.COLOR_BGR2GRAY)
        
        features = cv2.SIFT_create()
        kpts1, des1 = features.detectAndCompute(img_gray_1,None)
        kpts2, des2 = features.detectAndCompute(img_gray_2,None)
        img_gray_1 = cv2.drawKeypoints(img_gray_1, kpts1, 0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        img_gray_2 = cv2.drawKeypoints(img_gray_2, kpts2, 0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        output = cv2.drawMatches(img_gray_1, kpts1, img_gray_2, kpts2, matches[:40], None,matchColor =(0,255,255),flags = 2)
        cv2.imshow('img',output)
        cv2.imwrite('Match_keypoints.png',output)
        cv2.waitKey(0)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
  