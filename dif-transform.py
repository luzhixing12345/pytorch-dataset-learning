

from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms

def change_dim(pic):
    '''change dimension from [C H W] to [H W C]'''
    return pic.permute(1,2,0)

size = (400,400)

transform_method = [
    None,
    transforms.RandomCrop(size),
    transforms.CenterCrop(size),
    transforms.RandomResizedCrop(size),
    #transforms.FiveCrop(size), #5-tuple
    #transforms.TenCrop(size),  #10-tuple
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomRotation(30),
    transforms.Grayscale(3),
    transforms.ColorJitter(5),
    ]
plt.figure(figsize=(8, 8))
origin_image = read_image('./training-set/bird/01791f41a341270bbdb1b4c65eabfc41.jpeg')
origin_image = transforms.Resize((600,600))(origin_image)
#print(origin_image.shape)
plt.subplot(331)
plt.imshow(change_dim(origin_image))

for i in range(2,len(transform_method)+1):
    print(i)
    plt.subplot(3,3,i)
    plt.imshow(change_dim(transform_method[i-1](origin_image)))
plt.show()
