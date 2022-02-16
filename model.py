

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class NeuralNetwork_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=5)
        self.conv2 = nn.Conv2d(15, 20, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(360*512, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10,2)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view([-1, 360*512])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 
    
    
class NeuralNetwork_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*400*400, 400),
            nn.ReLU(),
            nn.Linear(400,100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50,2),
            nn.ReLU()
        )

    def forward(self, x):
        #print(x.shape)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits   
    
    

class Resnet18(nn.Module):
    '''
    预训练模型下载地址
    https://www.cnblogs.com/ywheunji/p/10605614.html
    '''
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.fc = nn.Linear(1000,2)
        
        self.model_weight_path = "model_weights/resnet18-5c106cde.pth"
        
        self.model.load_state_dict(torch.load(self.model_weight_path), strict=False)
        
    def forward(self,x):
        x = self.model(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

def build_model(model_type = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using {device} to calculate')
    
    CNN_type = [
        NeuralNetwork_conv,
        NeuralNetwork_linear,
        Resnet18
    ]
    
    model = CNN_type[model_type]().to(device)
    print(f'using {model.__class__.__name__}')
    return model

