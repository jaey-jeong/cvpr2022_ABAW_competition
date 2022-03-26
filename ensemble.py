import os
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from collections import OrderedDict
from networks.dan_yg import DAN_ab
from tqdm import tqdm
#from natsorted import natsorted
path = "./checkpoints/"
test_csv_path = "../src/test1.csv"
img_path = "../cropped_aligned/"

weight_list = os.listdir(path)
img_folder_list = os.listdir(img_path)

print(img_folder_list)
print(weight_list)


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
            model = DAN_ab(num_head=4, num_class=8)

            weight = path+model_name
            checkpoint = torch.load(weight)
            #model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            if isinstance(model, nn.DataParallel):
                model.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            else:  
                state_dict = checkpoint['model_state_dict']
                new_state_dict = OrderedDict() 
                for k, v in state_dict.items(): 
                    name = k[7:]
                    new_state_dict[name] = v 
                model.load_state_dict(new_state_dict)
            model.to(self.device)
            model.eval()    
            model_ = model
            
            self.all_models.append(model_)
            print('>loaded %s' % model_name)

        return self.all_models

    def fit(self, img):
        

        with torch.set_grad_enabled(False):
            img = img.to(self.device)
            targets = targets.to(self.device)
            
            for i in self.all_models:
                out, _, _ = self.model(img)
                outs+=out

            _, pred = torch.max(outs,1)
            index = pred

            return index

if __name__ == "__main__":
    #data = load_data()
    #model = load_all_models(weight_list)
    #print(len(model))
    model = Model()
    model_ = model.load_all_models(weight_list)
    df = pd.read_csv(test_csv_path)
    csvs, frames = df['name'].tolist(), df['frame'].tolist()
    for name, frame in tqdm(zip(csvs,frames)):
        print(str(name))
        print(len(str(frame)))
        y = []
        
        for i in tqdm(range(0,frame)):
            num = str(i+1)
            while(5>len(num)):
                num = "0"+num
            num=num+".jpg"
            #print(path+name+"/"+num)
            try:
                img = Image.open(path+name+"/"+num)
                #pred = 0
                model.eval()
                
                resize = transforms.Resize((224, 224))
                convert_tensor = transforms.ToTensor()
                img = resize(img)
                img = convert_tensor(img)
                img = img
                predict= model(img)
                #predict.axis(0)
                predict = predict.numpy()
                
                y.append(predict[0])
            except OSError as e:
                pred=-1
                print("img is not found")
                y.append(pred)
        csv = pd.DataFrame(data={'label':y})
        csv.to_csv(name+".csv",index=False)
    print(csvs, frames)
