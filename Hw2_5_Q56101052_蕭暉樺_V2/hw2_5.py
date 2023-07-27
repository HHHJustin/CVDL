from UI import Ui_mainWindow
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
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import datasets
from torch import nn
from torchsummary import summary
import PIL
# Dataset setting

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor()])
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
path_train = "Dataset_OpenCvDl_Hw2_Q5/training_dataset"
path_test = "Dataset_OpenCvDl_Hw2_Q5/inference_dataset"
TRAIN = Path(path_train)
TEST = Path(path_test)
train_data = datasets.ImageFolder(TRAIN,transform=train_transforms)
test_data = datasets.ImageFolder(TEST,transform=test_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, num_workers=4,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,  num_workers=1,shuffle=False)
classes = ('cat','dog')
model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 1)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.load_img_path)
        self.ui.pushButton_2.clicked.connect(self.show_img)
        self.ui.pushButton_3.clicked.connect(self.show_distribution)
        self.ui.pushButton_4.clicked.connect(self.show_structure)
        self.ui.pushButton_5.clicked.connect(self.show_compare)
        self.ui.pushButton_6.clicked.connect(self.inference)
        
    def load_img_path(self):
        self.img_filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
    
    def show_img(self):
        dataiter = iter(test_loader)
        cat_id = -1
        dog_id = -1
        for i in range(len(test_loader)):
            x,y = dataiter.next()
            if(y.item() == 0 and cat_id==-1):
                cat_id = i
                cat_img = x[0]
            if(y.item() == 1 and dog_id==-1):
                dog_id = i
                dog_img = x[0]
            if(cat_id!=-1 and dog_id!=-1):
                break
        cat_img = cat_img.permute(1,2,0).numpy()
        dog_img = dog_img.permute(1,2,0).numpy()
        plt.subplot(1,2,1)
        plt.title('Cat')
        plt.imshow(cat_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,2,2)
        plt.title('Dog')
        plt.imshow(dog_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    def show_distribution(self):
        img = cv2.imread("Figure_1.png")
        cv2.imshow("Class Distribution",img)

        ## Count

        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # total_cat = 0
        # total_dog = 0
        # dataiter = iter(train_loader)
        # for i in range(len(train_loader)):
        #     _,y = dataiter.next()
        #     y = y.to(device)
        #     total_cat+=(y==0).sum()
        #     total_dog+=(y==1).sum()
    
        # total_cat = total_cat.item()
        # total_dog = total_dog.item()
        
        # plt.ylabel("Number of image")
        # plt.title("Class Distribution")
        # cat = plt.bar('cat',total_cat)
        # dog = plt.bar('dog',total_dog)
        # cat_height = cat[0].get_height()
        # dog_height = dog[0].get_height()
        # plt.text(cat[0].get_x()+cat[0].get_width()/2.,cat_height,"%d" % int(cat_height),ha="center",va = "bottom",)
        # plt.text(dog[0].get_x()+dog[0].get_width()/2.,dog_height,"%d" % int(dog_height),ha="center",va = "bottom",)
        # plt.show()

    def show_structure(self):
        # print(model)
        print(summary(model,input_size=(3,224,224)))

    def show_compare(self):
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Comparison")
        bce_result = plt.bar('Binary Cross Entropy',95.11)
        focal_result = plt.bar('Focal loss',91.72)
        bce_height = bce_result[0].get_height()
        focal_height = focal_result[0].get_height()
        plt.text(bce_result[0].get_x()+bce_result[0].get_width()/2.,bce_height,"%.2f" % float(bce_height),ha="center",va = "bottom",)
        plt.text(focal_result[0].get_x()+focal_result[0].get_width()/2.,focal_height,"%.2f" % float(focal_height),ha="center",va = "bottom",)
        plt.show()

    def inference(self):
        classes = ('cat','dog')
        model = torch.load('model_with_BCE_loss.pt')
        model = model.to(device)
        model.eval()
        img = PIL.Image.open(self.img_filename)
        test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        input_img = test_transforms(img).to(device).unsqueeze(0)
        output = model(input_img)
        sigmoid = nn.Sigmoid()
        output = sigmoid(output)
        if(output[0].item()>0.5):
            prediction = classes[1]
        else:
            prediction = classes[0]
        display_img = cv2.imdecode(np.fromfile(self.img_filename,dtype=np.uint8),-1)
        display_img = cv2.resize(display_img,(224,224), interpolation=cv2.INTER_AREA)
        h,w,c = display_img.shape
        bytesPerline = 3 * w
        self.qimg = QImage(display_img, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.label_2.setText("Prediction:  " + prediction)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()