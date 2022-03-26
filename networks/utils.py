import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

 
def val_accuracy(output, target, p_, t_, f1, running_accuracy, total):

    _, pred = torch.max(output, 1)
    pred = torch.nn.functional.one_hot(pred, num_classes=8).float()
    tmp = f1_score(target.cpu(), pred.cpu(), average="macro")
    #print()
    #print("#######################################################")
    #print(tmp)
    #print(f1)
    #f1_ = f1
    f1 = np.append(f1, 100*tmp)
    #f1 = f1_
    #f1__ = np.sum(f1_)/8
    #print(f1_)
    #print(f1__)
    
    #print(f1)
    #print("#######################################################")
    print()
    
    for p, t in zip(pred, target) :
        p_.append(p.argmax().cpu())
        t_.append(t.argmax().cpu())
       
    correct = accuracy_score(target.cpu(), pred.cpu())
    #print(f1)
    running_accuracy += correct
    total += target.size(0)
    res = [p_, t_]

    return running_accuracy, total, res, f1

def accuracy(output, target, f1, running_accuracy, total):

    _, pred = torch.max(output, 1)
    
    
    pred = torch.nn.functional.one_hot(pred, num_classes=8).float()
    tmp = f1_score(target.cpu(), pred.cpu(), average="macro")
    f1 = np.append(f1, 100*tmp)
    #print(f1_)
    #print(f1)
    #print(f1_)
    correct = accuracy_score(target.cpu(), pred.cpu())
    #print(f1)
    running_accuracy += correct
    total += target.size(0)
    
    return running_accuracy, total, f1   



def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
