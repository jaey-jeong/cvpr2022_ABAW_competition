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
from networks.dan_yg import DAN_ab
from sklearn.metrics import balanced_accuracy_score, confusion_matrix ,f1_score
from collections import OrderedDict
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')
 
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
            model = DAN_ab(pretrained=True,num_head=4, num_class=8)

            weight = path+model_name
            checkpoint = torch.load(weight)
            #model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            print(weight)
            if "bagging" in weight: # GPU 병렬사용 적용 
                print("bagging")
                model.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            else: # GPU 병렬사용을 안할 경우 
                state_dict = checkpoint['model_state_dict']
                new_state_dict = OrderedDict() 
                for k, v in state_dict.items(): 
                    name = k[7:] # remove `module.` ## module 키 제거 
                    new_state_dict[name] = v 
                model.load_state_dict(new_state_dict)
            model.to(self.device)
            model.eval()    
            model_ = model
            
            self.all_models.append(model_)
            print('>loaded %s' % model_name)

        return self.all_models

    def fit(self, img,targets):
        
        outs = None
        with torch.set_grad_enabled(False):
            img = img.to(self.device)

            targets = targets.to(self.device)
            
            for model in self.all_models:
                out, _, _ = model(img)
                if(outs == None):
                    #print("들어왔니?")
                    outs=out
                else:
                    #print("너두?")
                    outs+=out

            _, pred = torch.max(outs,1)
            index = pred

            return index ,targets
'''
def plot_confusion_matrix(cm, plt_name, target_names=None, cmap=None, normalize=True, labels=True, title="DAN" +' confusion matrix'):
    fig1 = plt.gcf()
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.savefig("../cm/base/cm_"+str(plt_name)+".png")
    plt.savefig("../cm/cm_"+str(plt_name)+".png")
    print("PLT save ..")
    plt.close()
def split_train_val(path) :
    tmp = []
    folder = os.listdir(path)
    folder = natsorted(folder)
    for folder_ in folder :
        label = int(folder_)
        print(label)
        folder_ = os.listdir(path+"/"+folder_)
        
        
        folder_ = natsorted(folder_)
        for folder__ in folder_:
            tmp.append((path+"/"+str(label)+"/"+folder__,label))
    return tmp 
'''
def plot_confusion_matrix(cm, plt_name, target_names=None, cmap=None, normalize=True, labels=True, title="DAN" +' confusion matrix'):
    fig1 = plt.gcf()
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.savefig("../cm/base/cm_"+str(plt_name)+".png")
    plt.savefig("../cm/yg/cm_"+str(plt_name)+".png")
    print("PLT save ..")
    plt.close()
if __name__ == "__main__":
    '''
    args = parse_args()

    model = Model()

    image = args.image
    paths = split_train_val(image)
    p_=[]
    t_=[]
    for (image_path,label) in tqdm(paths):
        assert os.path.exists(image_path)
        index, label_pred = model.fit(image_path)
        targets = torch.eye(8)[label]
        pred = torch.eye(8)[label_pred]
        p_.append(pred)
        t_.append(targets)
    cm = confusion_matrix(p_,t_)
    plot_confusion_matrix(cm,"Training_"+"_cm", normalize = False, target_names = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])
    # print(f'emotion label: {label_pred}')
    '''
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
    path = "./strong_weak/"
    weight_list = os.listdir(path)
    model = Model()
    model_ = model.load_all_models(weight_list)
    val_dataset = datasets.ImageFolder("../image/affwild2/val", transform = data_transforms)    # loading statically
    # if args.num_class == 7:   # ignore the 8-th class 
    #     idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
    #     val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 1024,
                                               num_workers = 1,
                                               shuffle = False,  
                                               pin_memory = True)
    temp_exp_target = []
    temp_exp_pred = []
    iter_cnt = 0
    # image = args.image
    # assert os.path.exists(image)
    dataname = ["Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger","Other"]
    expr_x=["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"]
    p_=[]
    t_=[]
    for imgs, targets in tqdm(val_loader):
        # for a in targets:
        #     for i in range(len(dataname)):
        #         if(expr_x[a] == dataname[i]):
        #             a=i
        iter_cnt += 1
        predicts,targets = model.fit(imgs,targets)
        #print(predicts)
        #print(targets)
        for p, t in zip(predicts, targets) :
            p_.append(p.cpu())
            t_.append(t.cpu())
        
    f1=[]
    
    #print(temp_exp_pred)
    temp_exp_pred = np.array(p_)
    temp_exp_target = np.array(t_)
    temp_exp_pred = torch.eye(8)[temp_exp_pred]
    temp_exp_target = torch.eye(8)[temp_exp_target]
    for i in range(0,8):
        #print(temp_exp_pred)
        exp_pred = temp_exp_pred[:,i]
        exp_target = temp_exp_target[:,i]
        #print(exp_pred,exp_target)    
        f1.append(f1_score(exp_pred,exp_target))
    print(f1)
    running_f1 =np.mean(f1)
    cm = confusion_matrix(t_, p_)    
    plot_confusion_matrix(cm, "ensemble"+"_cm", normalize = False, target_names = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"])
    print(f'f1 score: {running_f1}')

