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
from networks.va import Va
import networks.resnet as ResNet
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/affwild2/', help='affwild2 dataset path.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=8, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--data_flags', type=str, default="fulldata", help='dataset name.')
    parser.add_argument('--aug', type=str, default=True, help='enable or disable augmentation')
    parser.add_argument('--opt', type=str, default=True, help='SGD,ADAM,ADAMW optimization')
    parser.add_argument('--sch', type=str, default=True, help='enable or disable scheduler')
    parser.add_argument('--pretrained', type=str, default=True, help='use pretrained model')

    return parser.parse_args()
    

def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    #print(np.shape(vx), np.shape(vy))
    rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))+1e-10)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)+1e-10)
    return 1 - ccc

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    #print(np.shape(vx), np.shape(vy))
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def metric_for_VA(gt_V,gt_A,pred_V,pred_A):
    ccc_V, ccc_A = CCC_score(gt_V, pred_V), CCC_score(gt_A, pred_A)
    return ccc_V, ccc_A

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class RafDataSet(data.Dataset):
    def __init__(self, image, label, transform = None):
        
        self.transform = transform
        
        self.image = image
        self.label = label
        #self.label2 = label[1]       

    def __len__(self):
        
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image[idx]

        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image, dtype=np.uint8)

        label = self.label[idx]
        #label2 = self.label[idx][1]

        label = torch.from_numpy(np.array(label)).float()
        #label2 = torch.from_numpy(np.array(label2)).float()
        
        return image, label

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=2, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        #print(batch_size)
        #print(self.num_class)
        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1+num_head/var)
        else:
            loss = 0
            
        return loss


def run_training():
    args = parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    model_name = "Pretrained_"+str(args.pretrained)+"_VA_"+str(args.aug)+"_LR_"+str(args.lr)+"_FREEZE_"+str(args.freeze)+"_OPT_"+str(args.opt)+"_schedule_"+str(args.sch)+"_num_head_"+str(args.num_head)    
    
    
    model = DAN_ab(args.pretrained, num_head=4, num_class=2)
    #summary(model,input_size=(1024,3,224,224))
    #checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
    #model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    #print("model : ",nn.Sequential(*list(model.children())[:]))
    #####
    #N_IDENTITY = 8631 
    #include_top = True
    #resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
    #####

    
    ct=0
    ct_=0
    for param in model.parameters():
        #print
        # param.requires_grad = False
        ct+=1
    print("Model Layer Num : ", ct)
    for param in model.parameters():
        ct_+=1

        if args.freeze == "100" :
            #param.requires_grad = True
            ct_ = 0
        elif args.freeze == "50" :
            if ct_ < ct*0.5 :
                param.requires_grad = False
            else: break
        elif args.freeze == "0" :
             if ct_ < ct-3 :
                param.requires_grad = False
    # for param in model.parameters():
    #     #print
    #     #print(ct)
    #     ct_+=1
    #     if ct_ < 17 : #50 frezee
    #         #print("false")
    #         param.requires_grad = False
    #     else:
    #         break
    print("freeze Layer Num : ", ct_)
    model.to(device)
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(mode=None),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
        transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2))], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        
        ])


    print("Data Load ...")
    X_train, y_train, X_val, y_val = load_pickle(args.data_flags)
    print()
    print("x : ",np.shape(X_train))
    print("y : ",np.shape(y_train))
    print("Data is Loaded !! ...")
    
    print("Generate train data set")
    train_dataset = RafDataSet(X_train, y_train, transform = data_transforms)    
    
        
    print('Whole train set size:', train_dataset.__len__())
    '''sampler=ImbalancedDatasetSampler(train_dataset),'''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(mode=None),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   
    print("Generate val data set")
    val_dataset = RafDataSet(X_val, y_val, transform = data_transforms_val)  
   
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    print("Set Optimizer .. ")
    #criterion_cls = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.365, 0.975, 0.986, 0.988, 0.837, 0.891, 0.958, 0.365]))
    criterion_cls = CCC_loss
    #criterion_cls = FocalLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()
    #criterion_cls = criterion_cls.cuda()
    params = list(model.parameters()) + list(criterion_af.parameters())
    
    if args.opt == "ADAM" :
        optimizer = torch.optim.Adam(params,args.lr,weight_decay = 1e-4)
    elif args.opt == "ADAMW" :
        optimizer = torch.optim.AdamW(params,args.lr,weight_decay = 0)
    elif args.opt == "SGD" :
        optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 0, momentum=0.9)
    
    if args.sch == "True":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
   
    

    #best_acc = 0
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()
    print("Model to device ..")
    print(count_parameters(model))


    #tmp_V_prob, tmp_A_prob, tmp_V_label, tmp_A_label = [], [], [], []
    train_loss = []
    val_loss = []
    bp = 0
    val_ccc_v = []
    val_ccc_a = []
    train_ccc_v = []
    train_ccc_a = []
    train_P = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        iter_cnt = 0
        model.train()
        tmp_V_prob, tmp_A_prob, tmp_V_label, tmp_A_label = [], [], [], []
        
        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            imgs = imgs.float()
            targets_ = targets
            targets_ = targets_.to(device)
            
            #targets = targets.to(device)
            out,feat,heads = model(imgs)
            #print(out, targets_)
            #criterion_cls = criterion_cls(out,targets_).cuda()
            #criterion_cls = criterion_cls.cuda()
            loss =  criterion_cls(out,targets_) + 1* criterion_af(feat,targets_) + 1*criterion_pt(heads) #89.3 89.4
            
            

            loss.backward()
          
            nn.utils.clip_grad_norm_(params, max_norm=1) 
         
            optimizer.step()
            
            running_loss += loss
            
            #print(np.shape(out[:][0]))

            tmp_V_prob.append(out[:][0].cpu().detach().numpy())
            tmp_V_label.append(targets[:][0].cpu().detach().numpy())
            tmp_A_prob.append(out[:][1].cpu().detach().numpy())
            tmp_A_label.append(targets[:][1].cpu().detach().numpy())

            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2
            
        running_loss = running_loss/iter_cnt
        train_ccc_v.append(ccc_v)
        train_ccc_a.append(ccc_a)
        train_P.append(final_VA_score)
        train_loss.append(running_loss.cpu().detach().numpy())
        
        
        tqdm.write('[Epoch %d]  Loss: %.3f. P: %.5f. LR %.6f. ' % (epoch, running_loss,final_VA_score,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
           
            
            #t_=[]
            
            val_P = []
            tmp_V_prob, tmp_A_prob, tmp_V_label, tmp_A_label = [], [], [], []
            model.eval()
            for (imgs, targets) in tqdm(val_loader):
                imgs = imgs.to(device)
                imgs = imgs.float()
                targets_ = targets
                targets_= targets_.to(device)
                
        
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets_) + criterion_af(feat,targets_) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                
                tmp_V_prob.append(out[:][0].cpu().detach().numpy())
                tmp_V_label.append(targets[:][0].cpu().detach().numpy())
                tmp_A_prob.append(out[:][1].cpu().detach().numpy())
                tmp_A_label.append(targets[:][1].cpu().detach().numpy())

                ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
                final_VA_score = (ccc_v + ccc_a) / 2
            
    
            running_loss = running_loss/iter_cnt   
            if args.sch == "True":
                scheduler.step()
            val_ccc_v.append(ccc_v)
            val_ccc_a.append(ccc_a)
            val_P.append(final_VA_score)
            val_loss.append(running_loss.cpu().detach().numpy())
            
            tqdm.write("[Epoch %d]  Loss:%.3f  P :%.5f " % (epoch, running_loss, final_VA_score))
            
            
            if final_VA_score > bp:
                bp = final_VA_score
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints_VA', "FINETUNE_VA_epoch"+str(epoch)+"_bp"+str(bp)+".pth"))
                tqdm.write('Model saved.')
                #to_csv(model_name, final_VA_score, val_loss)
                #print("csv saved ..")
            
    

if __name__=="__main__":
    run_training()