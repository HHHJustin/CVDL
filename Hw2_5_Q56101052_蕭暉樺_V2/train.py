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
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import datasets
import torchvision
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch import nn

def main():
    # Dataset setting
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, 1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    batch_size = 32
    path_train = "Dataset_OpenCvDl_Hw2_Q5/training_dataset"
    path_valid = "Dataset_OpenCvDl_Hw2_Q5/validation_dataset"
    TRAIN = Path(path_train)
    VALID = Path(path_valid)
    train_data = datasets.ImageFolder(TRAIN,transform=train_transforms)
    valid_data = datasets.ImageFolder(VALID,transform=valid_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)
    lr = 0.01
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    trainSteps = len(train_loader.dataset) 
    valSteps = len(valid_loader.dataset)
    epochs = 40
    lossFn = BCEWithLogitsLoss()
    for e in range(0,epochs):
        model.to(device)
        model.train()
        if e == 30:
            lr = 0.001
            opt = torch.optim.Adam(model.parameters(),lr = lr)
        totalTrainLoss = 0
        totalValLoss = 0 
        trainCorrect = 0
        valCorrect = 0
        for (x,y) in train_loader:
            (x,y) = (x.to(device),y.to(device).unsqueeze(1).float())
            pred =  model(x)
            loss = lossFn(pred,y)
            # loss = torchvision.ops.sigmoid_focal_loss(pred,y,reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss.item()
            pred_out = torch.zeros([x.shape[0],1])
            pred_out[torch.where(pred>0.5)] = 1
            pred_out = pred_out.to(device)
            num_correct = (pred_out == y).sum()
            trainCorrect += num_correct.item()
            
        with torch.no_grad():
            model.eval()
            for (x,y) in valid_loader:
                (x,y) = (x.to(device) , y.to(device).unsqueeze(1).float())
                pred =  model(x)
                loss = lossFn(pred,y)
                # loss = torchvision.ops.sigmoid_focal_loss(pred,y,reduction="sum")
                pred_out = torch.zeros([x.shape[0],1])
                pred_out[torch.where(pred>0.5)] = 1
                pred_out = pred_out.to(device)
                totalValLoss += loss.item()
                num_correct = (pred_out == y).sum()
                valCorrect += num_correct.item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = (trainCorrect / trainSteps) * 100
        valCorrect = (valCorrect / valSteps) * 100
                
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}%".format(avgTrainLoss, trainCorrect ))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}%\n".format(avgValLoss, valCorrect))
        if(e%10==0):
            torch.save(model, 'model_with_BCE_loss.pt')
    torch.save(model, 'model_with_BCE_loss_final.pt')

if __name__ =="__main__":
    main()