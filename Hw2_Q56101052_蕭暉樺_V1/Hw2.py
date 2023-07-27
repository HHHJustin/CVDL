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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.load_video_path)
        self.ui.pushButton_2.clicked.connect(self.load_img_path)
        self.ui.pushButton_3.clicked.connect(self.load_file)
        self.ui.pushButton_4.clicked.connect(self.background_subtraction)
        self.ui.pushButton_6.clicked.connect(self.processing)
        self.ui.pushButton_7.clicked.connect(self.video_tracking)
        self.ui.pushButton_10.clicked.connect(self.perspective_transform)
        self.ui.pushButton_11.clicked.connect(self.img_reconstruction)
        self.ui.pushButton_12.clicked.connect(self.compute_RE)

    def load_video_path(self):
        self.video_filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
    
    def load_img_path(self):
        self.img_filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")

    def load_file(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Open folder", "./")

    #Q4
    def img_reconstruction(self):
        img_name = listdir(self.folder_path)
        img_list = []
        for i in range(len(img_name)):
            img_path = join(self.folder_path,img_name[i])
            img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            img = img[:,:,::-1]
            img_list.append(img)
        dataset = np.array(img_list)
        dataset = dataset.reshape(-1,img.shape[0]*img.shape[1]*img.shape[2])
        pca = PCA(n_components=15,svd_solver="randomized", whiten=True)
        pca.fit(dataset)
        X_pca = pca.transform(dataset)
        X_new = pca.inverse_transform(X_pca).astype(int)
        self.img_h,self.img_w = img.shape[0],img.shape[1]
        self.output = []
        self.ori = []

        for j in range(len(img_name)):
            if j < 15:
                output_ = X_new[j].reshape(img.shape[0],img.shape[1],img.shape[2]).astype(int)
                self.output.append(output_)
                plt.subplot(4,15,j+1)
                plt.imshow(output_)
                plt.xticks([])
                plt.yticks([])
            else:
                output_ = X_new[j].reshape(img.shape[0],img.shape[1],img.shape[2]).astype(int)
                self.output.append(output_)
                plt.subplot(4,15,j+16)
                plt.imshow(output_)
                plt.xticks([])
                plt.yticks([])
        for j in range(len(img_name)):
            if j < 15:
                ori_ = img_list[j]
                self.ori.append(ori_)
                plt.subplot(4,15,j+16)
                plt.imshow(ori_)
                plt.xticks([])
                plt.yticks([])
            else:
                ori_ = img_list[j]
                self.ori.append(ori_)
                plt.subplot(4,15,j+31)
                plt.imshow(ori_)
                plt.xticks([])
                plt.yticks([])
        plt.show()
        

    def compute_RE(self):
        reconstruction_error = []
        for i in range(30):
            self.output[i] = cv2.cvtColor(self.output[i].astype(np.float32),cv2.COLOR_RGB2GRAY)
            self.ori[i] = cv2.cvtColor(self.ori[i].astype(np.float32),cv2.COLOR_BGR2GRAY)
            reconstruction_error.append((np.sum((self.output[i]-self.ori[i])**2))**0.5)
        print(reconstruction_error)

    #Q3
    def perspective_transform(self):
        cap = cv2.VideoCapture(self.video_filename)
        aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        img = cv2.imdecode(np.fromfile(self.img_filename,dtype=np.uint8),-1)
        h,w = img.shape[0],img.shape[1]
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
                for i in range(len(ids)):
                    if ids[i] == 1:
                        r_1 = corners[i][0][0]
                    elif ids[i] == 2:
                        r_2 = corners[i][0][1]
                    elif ids[i] == 3:
                        r_3 = corners[i][0][2]
                    elif ids[i] == 4:
                        r_4 = corners[i][0][3]
    
                pts_dst = np.array([r_1,r_2,r_4,r_3])
                pts_src = np.array([[(0,0),(w-1,0),(0,h-1),(w-1,h-1)]],dtype=np.float32)

                homo,status = cv2.findHomography(pts_src,pts_dst)
                warped_image = cv2.warpPerspective(img,homo,(frame.shape[1],frame.shape[0]))
                print(warped_image.shape)
                frame = cv2.fillConvexPoly(frame,np.array([r_1,r_2,r_3,r_4]).astype(int),0,16)
                out_img = cv2.add(frame,warped_image[:,:,0:3])
                cv2.imshow('frame', out_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
 
    #Q2
    def processing(self):
        cap = cv2.VideoCapture(self.video_filename)
        ret, frame = cap.read()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        params.filterByCircularity = True
        params.minCircularity = 0.83
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)
        self.keypoints_list = []
        for i in range(len(keypoints)):
            self.keypoints_list.append([[int(keypoints[i].pt[0]),int(keypoints[i].pt[1])]])
            with_keypoints = cv2.line(frame,(int(keypoints[i].pt[0]),int(keypoints[i].pt[1]-6)),((int(keypoints[i].pt[0]),int(keypoints[i].pt[1]+6))),(0,0,255))
            with_keypoints = cv2.line(frame,(int(keypoints[i].pt[0]-6),int(keypoints[i].pt[1])),((int(keypoints[i].pt[0]+6),int(keypoints[i].pt[1]))),(0,0,255))
            with_keypoints = cv2.rectangle(frame,(int(keypoints[i].pt[0]-6),int(keypoints[i].pt[1]-6)),(int(keypoints[i].pt[0]+6),int(keypoints[i].pt[1]+6)),(0,0,255))
        cv2.imshow("Keypoints", with_keypoints)
        cv2.waitKey(0)
        cap.release()

    def video_tracking (self):
        cap = cv2.VideoCapture(self.video_filename)
        lk_params = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = np.array(self.keypoints_list,dtype=np.float32)
        color = (0,0,255)
        mask = np.zeros_like(old_frame)
        while (cap.isOpened()):
            ret, frame = cap.read()    
            if ret == True:
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)
                img = cv2.add(frame, mask)
                cv2.imshow('frame', img)
                key = cv2.waitKey(10)
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                # ESC
                if key == 27:
                    break
                
            else:
                break
        cap.release()

    # Q1
    def background_subtraction(self):
        cap = cv2.VideoCapture(self.video_filename)
        gray_frame_list = []
        i = 0
        while (cap.isOpened() and i < 25):
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray_frame_list.append(gray_frame)
            
            if ret == True:
                key = cv2.waitKey(1)
                # ESC
                if key == 27:
                    break
            else:
                break
            i += 1
        
        gray_frame_array = np.array(gray_frame_list)
        first_25_mean = gray_frame_array.mean(axis=0) 
        
        first_25_std = gray_frame_array.std(axis=0) 
        first_25_std[np.where(first_25_std[:,:]<10)] = 10
        
        cap.release()
        cap = cv2.VideoCapture(self.video_filename)
        while (cap.isOpened()):
            ret, frame = cap.read()    
            if ret == True:
                h,w, = frame.shape[0],frame.shape[1]
                mask = np.zeros([h,w,3])
                foreground = frame.copy()
                gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                  
                mask[np.where(abs(gray_frame[:,:] - first_25_mean[:,:]) > first_25_std[:,:] * 5)] = 255
                foreground[:,:,0][np.where(abs(gray_frame[:,:] - first_25_mean[:,:]) <= first_25_std[:,:] * 5)] = 0
                foreground[:,:,1][np.where(abs(gray_frame[:,:] - first_25_mean[:,:]) <= first_25_std[:,:] * 5)] = 0
                foreground[:,:,2][np.where(abs(gray_frame[:,:] - first_25_mean[:,:]) <= first_25_std[:,:] * 5)] = 0

                output = np.zeros([h,w*3,3])
                output[0:h,0:w,:] = frame[:,:,:]/255
                output[0:h,w:2*w,:] = mask
                output[0:h,2*w:3*w,:] = foreground/255
                cv2.imshow('output.mp4',output)
                # out.write(frame)
                key = cv2.waitKey(10)
                # ESC
                if key == 27:
                    break
            else:
                break
        cap.release()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()