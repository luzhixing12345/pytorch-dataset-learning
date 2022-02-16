pytorch中的数据集的制作与使用
===
本项目为[知乎文章]()的代码实现,如有疑问可留issue

## Install & Dependence
- python = 3.7
- pytorch = 1.10.2
- torchvision = 0.11.3 （低版本的没有`read_image`函数,可根据知乎文章所说调整图片读取方式）
- numpy 
- matplotlib 
- h5py 

## Dataset Preparation
所有图片都在`test-set`,`training-set`之中，`test.hdf5`,`train.hdf5`为压缩数据集

## Use
基本用法：
```python
python main.py
```
调整模型,数据集读入方式
- `M`: `0对应CNN(default)` `1对应LNN` `2-对应Resnet18`
- `L`: `0-load in h5py(default)`,`1-load in readimage`
- `test`: 加入此选项只测试模型不训练
使用`LNN`+读入图片导入
```python
python main.py -M 1 -L 1
```
使用`Resnet18`且测试
```python
python main.py -M 2 --test
```
## Pretrained model
下载训练好的模型放在`./model_weights`下
| Model | Download |
| ---     | ---   |
| Model-RNN | [download](https://github.com/learner-lu/pytorch-dataset-learning/releases/download/V1.0.0/NeuralNetwork_conv.pth) |
| Model-LNN | [download](https://github.com/learner-lu/pytorch-dataset-learning/releases/download/V1.0.0/NeuralNetwork_linear.pth) |
| Model-Resnet18 | [download](https://github.com/learner-lu/pytorch-dataset-learning/releases/download/V1.0.0/Resnet18.pth) |


## Directory Hierarchy
```
|—— create_h5py.py
|—— dataset.py
|—— dif-transform.py
|—— main.py
|—— model.py
|—— model_weights
|    |—— 模型保存在这里
|    |—— 下载的模型也放在这里
|—— pytorch-tutorial.py  # pytorch官网的教程，可直接运行
|—— test-set
|    |—— bird
|        |—— 612706af4fb052215447c466472e1394.jpeg
|        |—— 66138d1941439c5bc1cbf1c94a19aa0f.jpeg
|        |—— a87ff6bb8fec05c9527676a844dec788.jpeg
|        |—— b295ba3fbc48e225353780b07adf0802.jpeg
|        |—— ba86e9c05865bde7014db50b9c22e48a.jpeg
|        |—— c6977f982380fb4f86ba730102c75406.jpeg
|        |—— d9e2a7551da3dac10066d42140d7f571.jpeg
|        |—— ddd3b055c5d3077e6ade6046b42c57bf.jpeg
|        |—— e439e2d54c2b6911c769beb5cfcf28a8.jpeg
|        |—— f295daf76a073e41284276fcaf03c334.jpeg
|    |—— flower
|        |—— c6bfa10e741759b76bbae64a77dbbf83.jpeg
|        |—— c7b0ba6f0649fc55132b25d34e50f65f.jpeg
|        |—— c86f4be37f0096c53096db9d23e45863.jpeg
|        |—— cb0b9d22bc770958731afa3170f45d95.jpeg
|        |—— cb5ee06378c5ca07be5cf85f3575c704.jpeg
|        |—— d0432f1c83ad65afd7b3a4d55d850d92.jpeg
|        |—— d0c88e3de35faead0d28aeb8e6cc8e27.jpeg
|        |—— e3413f158c38ff457a6484cadfcd0c1e.jpeg
|        |—— ed1c7f9559cb2ee623f9a53d2d9f9238.jpeg
|        |—— f13d062ad33b794f9a14c0279f0a295d.jpeg
|—— test.hdf5
|—— train.hdf5
|—— training-set
|    |—— bird
|        |—— 01791f41a341270bbdb1b4c65eabfc41.jpeg
|        |—— 0a78cb587f941d59489befa89285bb63.jpeg
|        |—— 130b9bfb1cd594ba7b12dae9df90f8a2.jpeg
|        |—— 246f43a069b6fef6577129bc6fa2e9cb.jpeg
|        |—— 29e71ce6aaebc9f06beffa9d9f01a21b.jpeg
|        |—— 2db22000281fc3cf7f87bb6ac6c0ceed.jpeg
|        |—— 34c6580b9b0c38854480e547f01d0795.jpeg
|        |—— 3e621180c51d70fa9dad8c427655367a.jpeg
|        |—— 46fd4de58c2266223e83a21370027180.jpeg
|        |—— 47ef371e4226ec6bba5b8092926a9bf0.jpeg
|        |—— 4f23da4fbf1f79be772ed4057938cbfd.jpeg
|        |—— 515c9cfdb994892d0d55e5518cb4a22d.jpeg
|        |—— 58f42d7c5abc1eedf16e61c21772ab67.jpeg
|        |—— 6368df61bc6be51ed776b7faa6b82570.jpeg
|        |—— 6da78b30fc5a1dc9dc3cc0d6e451baee.jpeg
|        |—— 717ca03970624b0c459426bc188de4cb.jpeg
|        |—— 71d9a4c1bf90904cceb0edc98c53b8a8.jpeg
|        |—— 73ee32e532f932a4bc34a16684df6f1c.jpeg
|        |—— 7f6879f668629a75457dd93feefc6209.jpeg
|        |—— 8bbde8be67c46cb07eab262224afeabe.jpeg
|    |—— flower
|        |—— 0754a441ffed0d494036e9f82ccc883c.jpeg
|        |—— 0bd85256a37087c086813bc7f5cd4398.jpeg
|        |—— 0e49e6273e41648fb7a4f1b804cc63a2.jpeg
|        |—— 0ecee7747d3388bd929b19df526fe7ab.jpeg
|        |—— 17e2ad2fed14d6d0b8d6999245c9ba4a.jpeg
|        |—— 38eb0ef9ef0b8289b3da07b5aecaa761.jpeg
|        |—— 400fa3440ecbb83f4a2f9e0330904570.jpeg
|        |—— 62ef9e022166f33bfad7401d6d9c8dbb.jpeg
|        |—— 71ebd1a8848170a22abea3d7c5d9e5b3.jpeg
|        |—— 821c4074003640f103794a462c5abe8a.jpeg
|        |—— 8fb8d5004c26bd8edd2646f0ee8b90bf.jpeg
|        |—— 9892491da39cd63bb0c8444871e4da18.jpeg
|        |—— 9c3b079c3318c70031ba67381d00fc4a.jpeg
|        |—— a0a369ae79108320343e81b8cffebd0f.jpeg
|        |—— aa8e9760b2267d8ead1a4fae74a22e04.jpeg
|        |—— bca392d862b6c4dde9ba1b7ae73a1b6a.jpeg
|        |—— bcb3596678a42d2058f4d634ec756ac1.jpeg
|        |—— bd4dcd9b6e5a891c8a7c16a678de3851.jpeg
|        |—— bd9e1b9c4df5993554a61fb611c7df94.jpeg
|        |—— bff530f2deef67d18b34db7ddd4f61ee.jpeg
```
## Code Details
### Tested Platform
- software
  ```
  OS: Windows11
  Python: 3.7.1 (anaconda)
  PyTorch: 1.10.2
  ```
- hardware
  ```
  CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz
  GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
  ```
### Hyper parameters
```
optimizer: SGD lr= 1e-4,momentum=0.5
EPOCH: 100
loss function: CrossEntropyLoss 
```

