from ui import Ui_MainWindow
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
        self.ui.pushButton.clicked.connect(self.open_folder)
        self.ui.pushButton_4.clicked.connect(self.open_img)
        self.ui.pushButton_5.clicked.connect(self.find_intrinsic)
        self.ui.pushButton_8.clicked.connect(self.find_extrinsic)
        self.ui.pushButton_6.clicked.connect(self.find_distortion)
        self.ui.pushButton_7.clicked.connect(self.show_result)
        self.ui.pushButton_9.clicked.connect(self.show_words_on_board)
        self.ui.pushButton_10.clicked.connect(self.show_words_vertically)
        self.ui.pushButton_2.clicked.connect(self.open_imgL)
        self.ui.pushButton_3.clicked.connect(self.open_imgR)
        self.ui.pushButton_11.clicked.connect(self.show_result_3)
        
    def open_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Open folder", "./")
    
    """ 第一題 """
    def open_img(self):
        self.objp_1 = np.zeros((11*8,3), np.float32)
        self.objp_1[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        self.objpoints_1 = [] # 3d point in real world space
        self.imgpoints_1 = []
        
        folder = self.folder_path.split('/')[3] + '\\' + self.folder_path.split('/')[4]
        for i in range(15):
            data_name = str(i+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            self.gray_img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,corners = cv2.findChessboardCorners(self.gray_img_1,(11,8),None) 
            if ret == True:
                self.objpoints_1.append(self.objp_1)
                self.imgpoints_1.append(corners)
                cv2.drawChessboardCorners(img, (11,8), corners, ret)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
                cv2.imshow('Image', img)
                cv2.waitKey(500)
                
    def find_intrinsic(self):
        print("Intrinsic Matrix:")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_1, self.imgpoints_1, self.gray_img_1.shape[::-1], None, None)
        print(mtx)
    
    def find_extrinsic(self):
        img_num = self.ui.comboBox.currentText()
        print("Extrinsic Matrix:")
        number = int(img_num)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_1, self.imgpoints_1, self.gray_img_1.shape[::-1], None, None)
        extrisic_m = []
        for i in range(15):
            rotation_matrix = cv2.Rodrigues(rvecs[i])[0]
            ex_m = np.concatenate((rotation_matrix,tvecs[i]), axis = 1)
            extrisic_m.append(ex_m)
        print(extrisic_m[number])
        
    def find_distortion(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_1, self.imgpoints_1, self.gray_img_1.shape[::-1], None, None)
        print("Distortion:")
        print(dist)
    
    def show_result(self):
        folder = self.folder_path.split('/')[3] + '\\' + self.folder_path.split('/')[4]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_1, self.imgpoints_1, self.gray_img_1.shape[::-1], None, None)
        img_list = listdir(folder)
        for i in range(len(img_list)):
            data_name = str(i+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            dst = cv2.undistort(img,mtx,dist)
            concat = np.hstack([img,dst])
            concat = cv2.resize(concat,(1024,512),interpolation = cv2.INTER_AREA)
            cv2.imshow('Image', concat)
            cv2.waitKey(500)
            
    """第二題"""
    def show_words_on_board(self):
        self.objp_2 = np.zeros((11*8,3), np.float32)
        self.objp_2[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        self.objpoints_2 = [] # 3d point in real world space
        self.imgpoints_2 = []
        folder = self.folder_path.split('/')[3] + '\\' + self.folder_path.split('/')[4]
        fs = cv2.FileStorage(join(folder,'Q2_lib','alphabet_lib_onboard.txt'),cv2.FILE_STORAGE_READ)
        img_list = listdir(folder)
        for i in range(5):
            data_name = str(i+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            self.gray_img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,corners = cv2.findChessboardCorners(self.gray_img_2,(11,8),None) 
            if ret == True:
                self.objpoints_2.append(self.objp_2)
                self.imgpoints_2.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_2, self.imgpoints_2, self.gray_img_2.shape[::-1], None, None)
        for p in range(5):
            data_name = str(p+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            imgpoints,_ = cv2.projectPoints(self.objpoints_2[p],rvecs[p],tvecs[p],mtx,dist)
            msg = self.ui.textEdit.toPlainText()
            for i in range(len(msg)):
                if i == 0:
                    dy = 5
                    dx = 7
                elif i == 1:
                    dy = 5
                    dx = 4           
                elif i == 2:
                    dy = 5
                    dx = 1
                elif i == 3:
                    dy = 2
                    dx = 7
                elif i == 4:
                    dy = 2
                    dx = 4
                elif i == 5:
                    dy = 2
                    dx = 1
                    
                ch = fs.getNode(msg[i]).mat()
                draw_line = ch.shape[0]
                for j in range(draw_line):
                    points = ((ch[j][0][1] + dy) * 11 + ch[j][0][0] + dx , (ch[j][1][1]+dy) * 11 + ch[j][1][0]+dx)
                    cvt_point_1 = (int(imgpoints[int(points[0])][0][0]),int(imgpoints[int(points[0])][0][1]))
                    cvt_point_2 = (int(imgpoints[int(points[1])][0][0]),int(imgpoints[int(points[1])][0][1]))
                    img = cv2.line(img,(cvt_point_1[0],cvt_point_1[1]),(cvt_point_2[0],cvt_point_2[1]),(0,0,255),5)
                    
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
            cv2.imshow('image',img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
    def show_words_vertically(self):
        self.objp_2 = np.zeros((11*8,3), np.float32)
        self.objp_2[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        self.objpoints_2 = [] # 3d point in real world space
        self.imgpoints_2 = []
        folder = self.folder_path.split('/')[3] + '\\' + self.folder_path.split('/')[4]
        fs = cv2.FileStorage(join(folder,'Q2_lib','alphabet_lib_vertical.txt'),cv2.FILE_STORAGE_READ)
        img_list = listdir(folder)
        for i in range(5):
            data_name = str(i+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            self.gray_img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,corners = cv2.findChessboardCorners(self.gray_img_2,(11,8),None) 
            if ret == True:
                self.objpoints_2.append(self.objp_2)
                self.imgpoints_2.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints_2, self.imgpoints_2, self.gray_img_2.shape[::-1], None, None)
        for p in range(5):
            data_name = str(p+1) + '.bmp'
            path = join(folder,data_name)
            img = cv2.imread(path)
            imgpoints,_ = cv2.projectPoints(self.objpoints_2[p],rvecs[p],tvecs[p],mtx,dist)
            msg = self.ui.textEdit.toPlainText()
            for i in range(len(msg)):
                if i == 0:
                    dy = 5
                    dx = 7
                elif i == 1:
                    dy = 5
                    dx = 4           
                elif i == 2:
                    dy = 5
                    dx = 1
                elif i == 3:
                    dy = 2
                    dx = 7
                elif i == 4:
                    dy = 2
                    dx = 4
                elif i == 5:
                    dy = 2
                    dx = 1

                ch = fs.getNode(msg[i]).mat()
                draw_line = ch.shape[0]
                for j in range(draw_line):
                    points = ((ch[j][0][2] + dy) * 11 + ch[j][0][0] + dx , (ch[j][1][2] + dy) * 11 + ch[j][1][0] + dx)
                    cvt_point_1 = (int(imgpoints[int(points[0])][0][0]),int(imgpoints[int(points[0])][0][1]))
                    cvt_point_2 = (int(imgpoints[int(points[1])][0][0]),int(imgpoints[int(points[1])][0][1]))
                    img = cv2.line(img,(cvt_point_1[0],cvt_point_1[1]),(cvt_point_2[0],cvt_point_2[1]),(0,0,255),5)
                    
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
            cv2.imshow('image',img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows() 
        
    def open_imgL(self):
        self.filenameL, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.filenameL = self.filenameL.split('/')[3]+ '\\' +self.filenameL.split('/')[4] + '\\' + self.filenameL.split('/')[5]
        self.img_L = cv2.imread(self.filenameL)
        self.img_L_resize = cv2.resize(self.img_L, (int(self.img_L.shape[1]/4), int(self.img_L.shape[0]/4)), interpolation=cv2.INTER_AREA)
        self.img_L = cv2.cvtColor(self.img_L, cv2.COLOR_BGR2GRAY)
                 

    def open_imgR(self):
        self.filenameR, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.filenameR = self.filenameR.split('/')[3]+ '\\' +self.filenameR.split('/')[4] + '\\' + self.filenameR.split('/')[5]
        self.img_R = cv2.imread(self.filenameR) 
        self.img_R_resize = cv2.resize(self.img_R, (int(self.img_R.shape[1]/4), int(self.img_R.shape[0]/4)), interpolation=cv2.INTER_AREA)
        self.img_R = cv2.cvtColor(self.img_R, cv2.COLOR_BGR2GRAY)
                 
        
    def show_result_3(self):
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(self.img_L, self.img_R)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disparity', int(disparity.shape[1]/4), int(disparity.shape[0]/4))
        cv2.imshow('disparity', disparity)
        event = cv2.EVENT_LBUTTONDOWN
        cv2.namedWindow('img_L')
        cv2.namedWindow('img_R_dot')
        self.img_R_backpack = self.img_R_resize
        cv2.setMouseCallback('img_L', self.draw_circle)
        cv2.imshow('img_L',self.img_L_resize)
        cv2.imshow('img_R_dot',self.img_R_resize)
        cv2.waitKey(0)
     
    def draw_circle(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            img_R_dot = np.copy(self.img_R_resize)
            cv2.circle(img_R_dot, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('img_R_dot',img_R_dot)

        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()

