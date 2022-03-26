from calendar import EPOCH
import os
import argparse
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms, datasets
import itertools
from networks.expr import Expr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix ,f1_score
from collections import OrderedDict
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, help='Image file for evaluation.')
    parser.add_argument('--bagging',type=str,default="./checkpoints_AIHUB_whole_weak/" ,help='bagging pretrained models')
    return parser.parse_args()

class Model() :
    def __init__(self) :
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])   
        self.all_models = []
        
    def load_all_models(self, weight_list):
        

        for model_name in weight_list:
            # load model from file
            model = Expr(pretrained=True,num_head=4, num_class=8)
            
            weight = path+model_name
            checkpoint = torch.load(weight)
            print(weight)
            try:
                state_dict = checkpoint['model_state_dict']
                new_state_dict = OrderedDict() 
                for k, v in state_dict.items(): 
                    name = k[7:]
                    new_state_dict[name] = v 
                model.load_state_dict(new_state_dict)
                print("bagging")
                
                if ((self.device.type == 'cuda') and (torch.cuda.device_count()>1)):
                    print('Multi GPU activate')
                    model = nn.DataParallel(model)
                    model = model.cuda()
                model.to(self.device)
                model.eval()    
                model_ = model
                
                self.all_models.append(model_)
                print('>loaded %s' % model_name)
                
            except: 
                 print("bagging")
            
        return self.all_models

    def fit(self, img,targets):
        
        outs = None
        with torch.set_grad_enabled(False):
            img = img.to(self.device)

            targets = targets.to(self.device)
            
            for model in self.all_models:
                out, _, _ = model(img)
                if(outs == None):
                    
                    outs=out
                else:
                    
                    outs+=out

            _, pred = torch.max(outs,1)
            index = pred

            return index ,targets
def mean_f1(preds,targets):
    f1=[]
    temp_exp_pred = np.array(preds)
    temp_exp_target = np.array(targets)
    temp_exp_pred = torch.eye(8)[temp_exp_pred]
    temp_exp_target = torch.eye(8)[temp_exp_target]
    for i in range(0,8):
        
        exp_pred = temp_exp_pred[:,i]
        exp_target = temp_exp_target[:,i]
        
        f1.append(f1_score(exp_pred,exp_target))
    return np.mean(f1)

if __name__ == "__main__":
   
    args = parse_args()
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #         transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
        #     ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        #transforms.RandomErasing(),
    ])
    
    weight_list = os.listdir(args.path)
    print(weight_list)
    model = Model()

    model_ = model.load_all_models(weight_list)
    val_dataset = datasets.ImageFolder(args.val, transform = data_transforms)    # loading statically
    # if args.num_class == 7:   # ignore the 8-th class 
    #     idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
    #     val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 2048,
                                               num_workers = 1,
                                               shuffle = False,  
                                               pin_memory = True)
    temp_exp_target = []
    temp_exp_pred = []
    iter_cnt = 0
    dataname = ["Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger","Other"]
    expr_x=["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"]
    p_=[]
    t_=[]
    for imgs, targets in tqdm(val_loader):
        
        iter_cnt += 1
        predicts,targets = model.fit(imgs,targets)
        
        for p, t in zip(predicts, targets) :
            p_.append(p.cpu())
            t_.append(t.cpu())
        
    f1=[]
    
    #print(temp_exp_pred)
   
    running_f1=mean_f1(p_,t_)
    cm = confusion_matrix(t_, p_)    
    asd = path.replace("/","")
    print(f'f1 score: {running_f1}')

