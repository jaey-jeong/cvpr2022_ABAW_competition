from cProfile import label
from ctypes import c_void_p
from email.mime import image
import os
from pickletools import long1
from typing import Type
import warnings
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
#from torchinfo import summary
from datasets.load_data import load_pickle
from torchvision.datasets import ImageFolder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix ,f1_score
import torch.nn.functional as F
from torch.autograd import Variable
from networks.dan import DAN
from networks.expr import Expr
import networks.resnet as ResNet
from collections import OrderedDict
class TestSet(data.Dataset):
    def __init__(self, path_list, max_size,name, transform = None):
        self.path = path_list  
        self.size = max_size       
        self.name = name
        self.image = []
        self.transform = transform
    def __len__(self):
        
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        num = str(idx+1)
        while(5>len(num)):
            num = "0"+num
        num=num+".jpg"    
        try:
            image = Image.open(self.path+self.name+"/"+num)
            self.image = image
            if self.transform is not None:
                image = self.transform(image)
        
        except:
            image = -1
        
        return image

if __name__ == "__main__":
    df=pd.read_csv("../src/test1.csv")
    name = df["name"]
    frame = df["frame"]
    path = "../cropped_aligned/"
    model = Expr(pretrained=True,num_head=4, num_class=8)
    weight = "./checkpoints44/epoch4_f1_0.3283626029967336_PRETRAINEDYes_SCRATCH_AUG_True_LR_0.0001_FREEZE_100_OPT_ADAM_schedule_True_num_head_4_.pth"
    checkpoint = torch.load(weight)
    #model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    print(weight)
    if "bagging" in weight: 
        print("bagging")
        model.load_state_dict(checkpoint['model_state_dict'],strict=True) 
    else: 
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict() 
        for k, v in state_dict.items(): 
            a = k[7:]  
            new_state_dict[a] = v 
        model.load_state_dict(new_state_dict)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),

        ])    
    model.eval()
    for name, frame in tqdm(zip(name,frame)):
        y = []
        test_set = TestSet(path,frame,name,data_transforms)
        test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size = 4,
                                               num_workers = 1,
                                               shuffle = False,  
                                               pin_memory = True)
        for img in tqdm(test_loader):
            if(img ==-1):
                y.append(img)
            else:
                out,feat,heads = model(img)
                _, predict = torch.max(out, 1)
                predict = predict.argmax().numpy()
                print(predict)
                y.append(predict)
        csv = pd.DataFrame(y,columns=['neutral, anger, disgust, fear, happiness, sadness, surprise, other'])
        csv.to_csv(name+".csv",index=False)
    
           
        #img = Image.open(path+name+"/frame"+str(frame)+".jpg")
    