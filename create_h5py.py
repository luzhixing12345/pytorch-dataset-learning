

import h5py
import os
from torchvision import transforms
from torchvision.io import read_image
import torch
import numpy as np

def create_file():
    
    print('start creating hdf5 file')
    training_root = "training-set"
    test_root = "test-set"
    
    all_type = ["flower","bird"]
    
    train_file = h5py.File("train.hdf5","w")
    test_file = h5py.File("test.hdf5","w")
    
    transform = transforms.Compose([
                transforms.Resize((600,600)),
                transforms.CenterCrop((400,400)),
                transforms.ConvertImageDtype(torch.float64),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
    
    data_enhancement = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30)
        ])
    train_data = []
    train_target = []

    test_data = []
    test_target = []
    
    for root,dirs,files in os.walk(training_root):
        target = root.split('\\')[-1]
        if target in all_type :
            for file in files: 
                pic = read_image(os.path.join(root,file))
                pic = transform(pic)
                pic_ = data_enhancement(pic)
                
                pic = np.array(pic).astype(np.float64)
                pic_ = np.array(pic_).astype(np.float64)
                
                train_data.append(pic)
                train_data.append(pic_)
                #print(pic.shape)
                train_target.append(target.encode())
                train_target.append(target.encode())

    
    train_file.create_dataset("image",data = train_data)
    train_file.create_dataset("target",data = train_target)  
            
    train_file.close()    
    print('finish training file')   
                
    for root,dirs,files in os.walk(test_root):
        target = root.split('\\')[-1]
        if target in all_type :
            for file in files: 
                pic = read_image(os.path.join(root,file))
                pic = transform(pic)
                pic_ = data_enhancement(pic)
                pic = np.array(pic).astype(np.float64)
                pic_ = np.array(pic_).astype(np.float64)
                
                test_data.append(pic)
                test_data.append(pic_)
                test_target.append(target.encode())
                test_target.append(target.encode())
    

    test_file.create_dataset("image",data = test_data)
    test_file.create_dataset("target",data = test_target)
    
    test_file.close()
    print("finish test file")
    
    