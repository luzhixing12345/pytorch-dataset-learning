
import torch
import h5py
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
import os
import numpy as np

class cls_dataset(Dataset):
    def __init__(self,root,all_type) -> None:
        super().__init__()
        self.root = root
        self.all_type = all_type
        self.transform = transforms.Compose([
                transforms.Resize((600,600)),
                transforms.CenterCrop((400,400)),
                transforms.ConvertImageDtype(torch.float64),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]), 
            ])
        self.data_enhancement = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30)
        ])
        self.set = []
        
        for root,dirs,files in os.walk(self.root):
            target = root.split('\\')[-1]
            if target in self.all_type:
                for file in files: 
                    pic = read_image(os.path.join(root,file))
                    pic = self.transform(pic)
                    pic_ = self.data_enhancement(pic)
                    #print(target)
                    if target == 'bird':
                        label = torch.tensor(0)
                    else:
                        label = torch.tensor(1)
                    information_1 = {
                        'image': pic,
                        'target':label
                    }
                    information_2 = {
                        'image': pic_,
                        'target':label
                    }
                    self.set.append(information_1)
                    self.set.append(information_2)
        
    def __getitem__(self, index):
        #print(self.set[index])
        return self.set[index]
    
    def __len__(self):
        return len(self.set)

class h5py_dataset(Dataset):
    
    def __init__(self,file_name) -> None:
        super().__init__()
        self.file_name = file_name
        

    def __getitem__(self, index):
        with h5py.File(self.file_name,'r') as f:
            if f['target'][index].decode() == "bird":
                target = torch.tensor(0)
            else :
                target = torch.tensor(1)
            return f['image'][index] , target
        
    def __len__(self):
        with h5py.File(self.file_name,'r') as f:
            return len(f['image'])

def load_datasets_readImage():
    
    training_root = "training-set"
    test_root = "test-set"
    
    all_type = ["flower","bird"]
    
    training_set = cls_dataset(root = training_root , all_type=all_type)
    test_set = cls_dataset(root = test_root, all_type= all_type)
    
    train_dataloader = DataLoader(training_set,batch_size=4)
    test_dataloader = DataLoader(test_set,batch_size=4)
    
    return train_dataloader,test_dataloader
  
  
  
    
def load_datasets_h5py():
    
    train_file = "./train.hdf5"
    test_file = "./test.hdf5"
    
    training_set = h5py_dataset(train_file)
    test_set = h5py_dataset(test_file)
    
    train_dataloader = DataLoader(training_set,batch_size=4)
    test_dataloader = DataLoader(test_set,batch_size=4)
    
    return train_dataloader,test_dataloader

