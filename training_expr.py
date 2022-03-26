from cProfile import label
import os
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
from networks.expr import Expr
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


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
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
class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    

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
    model_name = "AIHUB_STRONG_PRETRAINED"+args.pretrained+"_SCRATCH_AUG_"+str(args.aug)+"_LR_"+str(args.lr)+"_FREEZE_"+str(args.freeze)+"_OPT_"+str(args.opt)+"_schedule_"+str(args.sch)+"_num_head_"+str(args.num_head)    
    
    
    model = Expr(args.pretrained, num_head=4, num_class=8)
    
    
    model.to(device)
    
    data_transforms = transforms.Compose([
       
        transforms.Resize((224,224)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        
        ])


   
    train_dataset = ImageFolder(args.data_path+"/train",transform =data_transforms)
        
    print('Whole train set size:', train_dataset.__len__())
    '''sampler=ImbalancedDatasetSampler(train_dataset),'''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler =ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   
    print("Generate val data set")
   
    val_dataset = ImageFolder(args.data_path+"val",transform =data_transforms_val)
  
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    print("Set Optimizer .. ")
    #criterion_cls = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.365, 0.975, 0.986, 0.988, 0.837, 0.891, 0.958, 0.365]))
    criterion_cls = FocalLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()
    criterion_cls = criterion_cls.cuda()
    params = list(model.parameters()) + list(criterion_af.parameters())
    if args.opt == "ADAM" :
        optimizer = torch.optim.Adam(params,args.lr,weight_decay = 1e-4)
    elif args.opt == "ADAMW" :
        optimizer = torch.optim.AdamW(params,args.lr,weight_decay = 0)
    elif args.opt == "SGD" :
        optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 0, momentum=0.9)
    
    if args.sch == "True":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
   
    

    best_acc = 0
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()
    print("Model to device ..")
    print(count_parameters(model))
    train_f1 = []
    val_f1 = []
    train_acc= []
    val_acc = []
    train_loss = []
    val_loss = []
    bf1 = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        temp_exp_target = []
        temp_exp_pred = []
        running_f1=0.0
        
        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            imgs = imgs.float()
            targets_ =targets
            targets_=targets_.to(device)
            
            targets = torch.eye(8)[targets]
            targets = targets.to(device)
            out,feat,heads = model(imgs)
            loss = criterion_cls(out,targets_) + 1* criterion_af(feat,targets) + 1*criterion_pt(heads) #89.3 89.4
            
            loss.backward()
     
            nn.utils.clip_grad_norm_(params, max_norm=1) 
         

            running_loss += loss
            
            
            _, predicts = torch.max(out, 1)
            
            for i in range(predicts.shape[0]):                
                temp_exp_pred.append(predicts[i].cpu().numpy())
                temp_exp_target.append(targets[i].argmax().cpu().numpy())
           
            correct_num = torch.eq(predicts, targets.argmax(axis=1)).sum()
            correct_sum += correct_num
        
        f1=[]
       
        temp_exp_pred = np.array(temp_exp_pred)
        temp_exp_target = np.array(temp_exp_target)
        temp_exp_pred = torch.eye(8)[temp_exp_pred]
        temp_exp_target = torch.eye(8)[temp_exp_target]
        for i in range(0,8):
           
            exp_pred = temp_exp_pred[:,i]
            exp_target = temp_exp_target[:,i]
            
            f1.append(f1_score(exp_pred,exp_target))
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        running_f1 =np.mean(f1)
      
        print(type(acc))
        train_acc.append(acc.cpu().numpy())
        train_loss.append(running_loss.cpu().detach().numpy())
        train_f1.append(running_f1)
        
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. F1: %.5f. LR %.6f. ' % (epoch, acc, running_loss,running_f1,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []
            running_f1 = 0.0
            p_=[]
            t_=[]
            model.eval()
            
            for (imgs, targets) in tqdm(val_loader):
                imgs = imgs.to(device)
                imgs = imgs.float()
                targets_ =targets
                targets_=targets_.to(device)
                targets = torch.eye(8)[targets]
               
                targets = targets.to(device)
                
        
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets_) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
            
                correct_num  = torch.eq(predicts,targets.argmax(axis=1))
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                for p, t in zip(predicts, targets) :
                    p_.append(p.cpu())
                    t_.append(t.argmax().cpu())
               
                baccs.append(balanced_accuracy_score(targets.cpu().numpy().argmax(axis=1),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            if args.sch == "True":
                scheduler.step()
            
            
            
           
            running_f1= mean_f1(p_,t_)
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)
            
            val_acc.append(acc)
            val_loss.append(running_loss.cpu().detach().numpy())
            val_f1.append(running_f1)
            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f f1 :%.5f " % (epoch, acc, bacc, running_loss,running_f1))
            tqdm.write("best_acc:" + str(best_acc))
            #print(f1)
            
            if running_f1 > bf1 :
                bf1 = running_f1
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints_AIHUB_strong', "epoch"+str(epoch)+"_f1_"+str(bf1)+"_"+model_name+"_.pth"))
                tqdm.write('Model saved.')


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
if __name__=="__main__":
    run_training()