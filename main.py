

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import argparse,os
from create_h5py import create_file
from model import build_model
from dataset import load_datasets_h5py,load_datasets_readImage
device = "cuda" if torch.cuda.is_available() else "cpu"

def default_argument_parser():
    parser = argparse.ArgumentParser(description="pytorch-dataset-study")
    parser.add_argument('--test',action="store_true",help="only test the model")
    parser.add_argument('-M','--model-type', default=0,type=int)
    parser.add_argument('-L','--load-type', default=0,type=int)
    return parser

def change_dim(pic):
    '''change dimension from [C H W] to [H W C]'''
    return pic.permute(1,2,0)

def main(args):
    
    model = build_model(args.model_type)
    
    if args.load_type == 0:
        if not os.path.exists('train.hdf5') or not os.path.exists('test.hdf5'):
            create_file()
        training_dataloader ,test_dataloader = load_datasets_h5py()
        print('using h5py method to load')
    else:
        training_dataloader ,test_dataloader = load_datasets_readImage()
        print('using readImage method to load')
    
    optimizer = optim.SGD(model.parameters(),lr= 1e-4,momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()
    EPOCH = 100
    
    loss_all = []
    if not args.test:
        print('start training')
        for epoch in range(EPOCH):
            print(f'\n-----------epoch {epoch}-----------')
            loss = train(model,training_dataloader,optimizer,loss_fn,epoch=epoch)
            loss_all.append(loss)
            test(model,test_dataloader)
        
        plt.plot(loss_all)
        plt.savefig(f"model_weights/{model.__class__.__name__}.png")
        plt.show()
        plt.close()
        
        torch.save(model.state_dict(), f"model_weights/{model.__class__.__name__}.pth")
        print("Saved PyTorch Model State to model.pth")
    
    model = build_model(args.model_type)
    model.load_state_dict(torch.load(f"model_weights/{model.__class__.__name__}.pth"))
    labels = {0:'bird',1:'flower'}
    model.eval()
    plt.figure(figsize=(8, 4))
    for id,data in enumerate(test_dataloader):
        
        if isinstance(data,list):
            image = data[0].type(torch.FloatTensor).to(device)
            #target = data[1].to(device)
        elif isinstance(data,dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            #target = data['target'].to(device)
        else :
            raise TypeError
        
        plt.title("image-show")
        with torch.no_grad():
            output =nn.Softmax(dim=1)(model(image))
            
            pred = output.argmax(dim = 1).cpu().numpy()

            plt.ion()
            for i in range(1,5):
                plt.subplot(1,4,i)
                plt.title(labels[pred[i-1]])
                plt.imshow(change_dim(image[i-1].cpu()))
            plt.pause(3)
            plt.show()
            

def train(model,train_dataloader,optimizer,loss_fn,epoch):
    model.train()
    
    loss_total = 0
    for _, data in enumerate(train_dataloader):

        if isinstance(data,list):
            image = data[0].type(torch.FloatTensor).to(device)
            target = data[1].to(device)
        elif isinstance(data,dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            target = data['target'].to(device)
        else :
            print(type(data))
            raise TypeError
        #print(target)
        optimizer.zero_grad()
        output = model(image)
        #print(output)

        loss = loss_fn(output, target)
        loss_total+=loss.item()
        
        loss.backward()
        optimizer.step()
    #exit(0)
    print(f'{round(loss_total,2)} in epoch {epoch}')
    return loss_total

def test(model,test_dataloader):
    model.eval()
    correct = 0

    for _ , data in enumerate(test_dataloader):

        if isinstance(data,list):
            image = data[0].type(torch.FloatTensor).to(device)
            target = data[1].to(device)
        elif isinstance(data,dict):
            image = data['image'].type(torch.FloatTensor).to(device)
            target = data['target'].to(device)
        else :
            raise TypeError
        
        with torch.no_grad():
            output = model(image)
            pred = nn.Softmax(dim=1)(output)

        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    
    print(f'accurency = {correct}/{len(test_dataloader)*4} = {correct/len(test_dataloader)/4}')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

