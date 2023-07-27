from UI import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from os.path import join
from os import listdir
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchsummary import summary
from PIL import Image
from tensorboardX import SummaryWriter 
from torch.utils.data import DataLoader
import cv2
import math

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        (self.traindata,self.valdata) = random_split(self.trainset,[40000,10000],generator=torch.Generator().manual_seed(42))
        self.setup_control()
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.model = torchvision.models.vgg19(num_classes=10)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    def setup_control(self):
        self.ui.pushButton_2.clicked.connect(self.show_train_img)
        self.ui.pushButton_3.clicked.connect(self.show_model)
        self.ui.pushButton_4.clicked.connect(self.show_augmentation)
        self.ui.pushButton_5.clicked.connect(self.show_AC_Loss)
        self.ui.pushButton_6.clicked.connect(self.inference)
        
    def show_train_img(self):
        imgs_no = random.sample(range(0,40000), 9)
        fig = plt.figure()
        fig.canvas.manager.set_window_title("Train Image")
        for i in range(len(imgs_no)):
            img, label = self.traindata[imgs_no[i]]
            img = img.permute(1,2,0)
            plt.subplot(3, 3, i+1)
            plt.title(self.classes[label])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        plt.tight_layout()
        plt.show()
    
    def show_model(self):
        VGG19 = self.model.to(self.device)
        print(summary(VGG19,(3,32,32)))
        
    def show_augmentation(self):
        img = Image.open('test.jpg').convert("RGB")
        plt.subplot(2,2,1)
        plt.title('origin')
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        
        transform_1 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(1)])
        img_1 = transform_1(img).permute(1,2,0).numpy()
        plt.subplot(2,2,2)
        plt.title('RandomHorizontalFlip')
        plt.imshow(img_1)
        plt.xticks([])
        plt.yticks([])
        
        transform_2 = transforms.Compose([transforms.ToTensor(),transforms.RandomCrop(512)])
        img_2 = transform_2(img).permute(1,2,0).numpy()
        plt.subplot(2,2,3)
        plt.title('RandomCrop')
        plt.imshow(img_2)
        plt.xticks([])
        plt.yticks([])
        
        transform_3 = transforms.Compose([transforms.ToTensor(),transforms.RandomGrayscale(1)])
        img_3 = transform_3(img).permute(1,2,0).numpy()
        plt.subplot(2,2,4)
        plt.title('RandomGrayscale')
        plt.imshow(img_3)
        plt.xticks([])
        plt.yticks([])
        plt.show()
     
    """訓練model用"""
    def train(self):
        writer = SummaryWriter()
        batch_size =32
        trainDataLoader = DataLoader(self.traindata, shuffle = True, batch_size=batch_size)
        valDataLoader = DataLoader(self.valdata, shuffle = True, batch_size=batch_size)
        model_VGG = self.model.to(self.device)
        learning_rate = 0.001
        opt = torch.optim.SGD(model_VGG.parameters(), lr=learning_rate)
        lossFn = nn.CrossEntropyLoss()
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
            }
        trainSteps = len(trainDataLoader.dataset) 
        valSteps = len(valDataLoader.dataset)
        print(self.device)
        for e in range(0,30):
            print(e)
            model_VGG.train()
            totalTrainLoss = 0
            totalValLoss = 0 
            trainCorrect = 0
            valCorrect = 0
            for (x,y) in trainDataLoader:
                
                (x,y) = (x.to(self.device),y.to(self.device))
                pred =  model_VGG(x)
                loss = lossFn(pred,y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                totalTrainLoss += loss.item() 
                _, pred_out = torch.max(pred,1)
                num_correct = (pred_out == y).sum()
                trainCorrect += num_correct.item()
                
            with torch.no_grad():
                model_VGG.eval()
                for (x,y) in valDataLoader:
                    (x,y) = (x.to(self.device) , y.to(self.device))
                    pred =  model_VGG(x)
                    totalValLoss += lossFn(pred, y) 
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            trainCorrect = (trainCorrect / trainSteps) * 100
            valCorrect = (valCorrect / valSteps) * 100
            
            H["train_loss"].append(avgTrainLoss)
            H["train_acc"].append(trainCorrect)
            writer.add_scalar('Accuracy', trainCorrect,e)
            writer.add_scalar('Loss',avgTrainLoss,e)
            H["val_loss"].append(avgValLoss)
            H["val_acc"].append(valCorrect)
            print("[INFO] EPOCH: {}/{}".format(e + 1, 30))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}%".format(avgTrainLoss, trainCorrect ))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}%\n".format(avgValLoss, valCorrect))
        writer.close()
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label = "Train_Loss")
        plt.title("Train_Loss")
        plt.xlabel("# Epoch")
        plt.ylabel("Loss")
        plt.savefig("Loss.png")
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_acc"], label = "Train_Accuracy")
        plt.title("Train_Accurancy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accurancy(%)")
        plt.savefig("Accuracy.png")
        torch.save(model_VGG, 'Save_model_VGG19.pt')
    
    def show_AC_Loss(self):
        img = cv2.imread('AC&Loss.png')
        cv2.imshow('Accuracy',img)
        cv2.waitKey(0)
    
    def inference(self):
        model = torch.load('Save_model_VGG19.pt').to(self.device).eval()
        test_num = random.randint(0,9999)
        x,y = self.testset[test_num]
        x_img = x.permute(1,2,0).numpy()
        x_img = cv2.resize(x_img, (32*10,32*10))
        format_img = np.zeros([1,x.shape[0],x.shape[1],x.shape[2]])
        format_img[0] = x
        format_img = torch.tensor(format_img,dtype = torch.float)
        m = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            format_img = format_img.to(self.device) 
            pred = model(format_img)
            pred = m(pred)
            confidence,predicted = pred.max(1)
            confidence = round(confidence.item(),2)
            predicted = predicted.cpu().numpy().astype(np.uint8)
        plt.title('Confidence:' + str(confidence*100) + '%' + '\n' + 'Prediction Label:' + self.classes[int(predicted)] )
        plt.imshow(x_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()