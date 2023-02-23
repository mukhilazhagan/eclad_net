from skimage import io, transform
from torch.utils.data import Dataset
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from definitions import MSE


class Logo_Dataset(Dataset):

    def __init__(self, img_list, class_list, transform=None):
        self.img_list = img_list
        self.class_list = class_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img = plt.imread(pd['Filename'][i])
        # class_name = plt.imread(pd['Classname'][i])
        # bbox = plt.imread(pd['Boundingbox'][i])
        img = self.img_list[idx]
        class_name = self.class_list[idx]
        sample = {'image': img, 'class_name': class_name}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
class Rescale(object):
    """ Rescaling image to given size - output_size
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, class_name = sample['image'], sample['class_name']
        #print(image.shape)
        #print(self.output_size)
        h, w = image.shape[:2]
        """
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size 
        """
        new_h =  self.output_size
        new_w =  self.output_size
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'class_name': class_name}
#class RandomCrop(object):
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, class_name = sample['image'], sample['class_name']

        """
        if len(image.shape)>2:
            image_gray = 0.2999*image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]
        else:
            image_gray = image
        """

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'class_name': torch.tensor(class_name)}
# Define NN
import torch.nn as nn
import torch.nn.functional as F
class GhostModule(nn.Module):
    
    # Ex : inp = 3,oup = 12, kernel_size = 5(same settings than classical network)
    def __init__(self,inp,oup,kernel_size=1,ratio=2,dw_size=3,stride=1,relu=True):
        super(GhostModule,self).__init__()
        self.oup = oup
    
        ### Compute channels for both primary and secondary based on ratio (s)
        # Ex : init_channels = 12*0.3 = 3.6 = 4
        init_channels = math.ceil(oup/ratio)
            
        # Ex : new_channels = 12-4 = 6
        # Ex : new_channels + init_channels = 12 = oup
        # Not : new_channels =  oup - init_channels beacause 
        # new_channels has to be divisible by init_channels (mandatory by pytorch)
        new_channels =  init_channels*(ratio-1)
        ## Primary standard convolution + MP + ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride),
            nn.ReLU(inplace=True)
        )
        
    
        ### Secondary depthwise convolution + ReLU
        ##  dw_size//2 to keep same Height and Width for the new Features Maps
        self.cheap_operation = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels)
        # groups allow to perform convolution only once for each matrix 

    def forward(self, x):
        # tic = time.perf_counter()
        x1 = self.primary_conv(x)
        # toc = time.perf_counter()
        # print(f"Classic Conv in {toc-tic:0.12f} seconds")        
                       
        # tic = time.perf_counter()
        x2 = self.cheap_operation(x1)
        # toc = time.perf_counter()
        # print(f"Cheap Conv in {toc-tic:0.12f} seconds")
        

        ### Stack Standard and Depthwise
        out = torch.cat([x1,x2], dim=1)
        
        ## Sometimes, to ensure that out_channels is divisble by groups (mandatory by pytorch)
        # cheap operation compute more feature map
        # Thats why it retrieve only the necessary number of layer, not all of them it's to much has been computed
        return out[:,:self.oup,:,:]
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # Original images are 32x32
        self.conv1 = nn.Conv2d(3, 12, 5)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(12, 24, 5)
        # Ghost Module
        #self.conv2 = GhostModule(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        ## Output dim of conv layer is 16(nb_channels)*5*5(remaining dim of pictures)
        # Linear(in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: Any | None = None)
        self.fc1 = nn.Linear(24*5*5, 32)
        ## Res or Cap ? Reason for last dim of two
        self.fc2 = nn.Linear(32, 2)
        ## SoftMax 
        self.sm1 = nn.Softmax(dim=1)
    
    ## x correspond to the image
    def forward(self, x):
        # Conv + ReLu + Pool (First Layer)
        # tic = time.perf_counter()
            
        ## Computeall in once if no conv plot required
        xtemp1_ = F.relu(self.conv1(x))
        x_temp1 = self.pool(xtemp1_)
        
        
        # Conv + ReLu + Pool (Second Layer)

        xtemp2_ = F.relu(self.conv2(x_temp1))
        x_temp2 = self.pool2(xtemp2_)
             
             
        # Ghost Module
        # x = self.pool2(self.conv2.forward(x))
        # -1 re arrange array regarding the second parameter
        ## Error "shape '[-1, 400]' is invalid for input of size 4096"
        x_ = x_temp2.view(-1, 24*5*5)
        x_ = F.relu(self.fc1(x_))
        # No relu for fc2 cause we use softMax to end up with probability for the class
        x_ = self.fc2(x_)
        x_ = self.sm1(x_)
        

        return x_

class GhostNet(nn.Module):
    def __init__(self,ratio1=2,ratio2=2):
        super(GhostNet, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # Original images are 32x32
        self.ghost1 = GhostModule(3,12,kernel_size=5,ratio=ratio1,dw_size=5,relu=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.ghost2 = GhostModule(12,24,kernel_size=5,ratio=ratio2,dw_size=5,relu=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        ## Output dim of conv layer is 16(nb_channels)*5*5(remaining dim of pictures)
        # Linear(in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: Any | None = None)
        self.fc1 = nn.Linear(24*5*5, 32)
        ## Res or Cap ? Reason for last dim of two
        self.fc2 = nn.Linear(32, 2)
        ## SoftMax 
        self.sm1 = nn.Softmax(dim=1)
    
    ## x correspond to the image
    def forward(self, x):
        # Conv + ReLu + Pool (First Layer)
        # tic = time.perf_counter()
        x_temp1_ = self.ghost1(x)
        x_temp1 = self.pool1(F.relu(x_temp1_))
        # toc = time.perf_counter()
        # print(f"Trained in {toc-tic:0.4f} seconds")
        # Conv + ReLu + Pool (Second Layer)
        x_temp2_= self.ghost2(x_temp1)
        x_temp2 = self.pool1(F.relu(x_temp2_))
        # x = self.pool2(self.conv2.forward(x))
        # -1 re arrange array regarding the second parameter
        ## Error "shape '[-1, 400]' is invalid for input of size 4096"
        x_ = x_temp2.view(-1, 24*5*5)
        x_ = F.relu(self.fc1(x_))
        # No relu for fc2 cause we use softMax to end up with probability for the class
        x_ = self.fc2(x_)
        x_ = self.sm1(x_)
        
        return x_
        