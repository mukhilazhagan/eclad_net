# %%
from __future__ import print_function, division
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from read_images import read_images

import os
import torch
import torchvision
import random
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

is_store_data = False
is_read_data = True
# %%
if is_read_data:
    img_list, class_list = read_images()

# %%
class_list
# %%
# Converting classes to onehotencoding
class_set = set(class_list)
class_int = [x for x in range(0,len(class_set))]

class_one_hot = [list(class_set).index(class_list[x]) for x in range(len(class_list))]
one_hot = torch.nn.functional.one_hot(torch.tensor(class_one_hot))
print(class_one_hot)
#class_one_hot = [x for item in class_list] 

# Problems where CrossEntropy Loss doesn't accept onehot value so reverting to class indices values
#class_list = np.asarray(class_one_hot).astype(np.float32)
class_list = np.asarray(class_one_hot)

# %%
print(type(class_list[1]))
print(type(img_list))
# %%

def store_data(is_store_true, img_list, class_list):
    if is_store_true:
        file_wr = open('img_data.pkl', 'wb')
        pickle.dump(img_list, file_wr)
        file_wr.close()

        file_wr_c = open('class_data.pkl', 'wb')
        pickle.dump(class_list, file_wr_c)
        file_wr_c.close()

        file_rd = open('img_data.pkl', 'rb')
        img_list = pickle.load(file_rd)
        file_rd.close()

        file_rd_c = open('class_data.pkl', 'rb')
        class_list = pickle.load(file_rd_c)
        file_rd_c.close()
    else:
        file_rd = open('img_data.pkl', 'rb')
        img_list = pickle.load(file_rd)
        file_rd.close()
        file_rd_c = open('class_data.pkl', 'rb')
        class_list = pickle.load(file_rd)
        file_rd_c.close()

    return img_list, class_list

#is_store_data = False
#img_list, class_list = store_data(is_store_data, img_list, class_list)

# %%
def test_data_load(img_list, class_list, idx):
    plt.imshow(img_list[idx])
    print("Image Size: ", img_list[idx].shape)
    print(class_list[idx])
    #print(img_list[idx])

test_data_load(img_list, class_list, 2)

# %%

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
        img = img_list[idx]
        class_name = class_list[idx]
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
        
# %%
transformed_dataset = Logo_Dataset(img_list, class_list  , transform = transforms.Compose([Rescale(32), ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


# %% Visualize Random Samples
dataiter = iter(dataloader)
sample_batched = dataiter.next()

for i in range(4):
    plt.figure(i)
    plt.imshow(sample_batched['image'][i].numpy().transpose((1, 2, 0)))
    plt.title(sample_batched['class_name'][i])
# %%

# Define NN
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas =(0.9,0.999))
# %%

# List Capturing Loss Curve
loss_capture = []

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = sample_batched['image'].double()
        labels = sample_batched['class_name']
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_capture.append(loss.item())
        if i_batch % 20 == 19:    #Print every 20 itr
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
plt.plot(loss_capture)
plt.title("Loss Curve")

# %%
# Tensorboard
 
# default `log_dir` is "runs" 
writer = SummaryWriter('runs/res_and_cap_exp')
dataiter = iter(dataloader)
batch_tb = dataiter.next()
img_grid = torchvision.utils.make_grid(batch_tb['image'])

# %%
# write to tensorboard
writer.add_image('four images', img_grid)
## At this stage Run in command Line tensorboard --logdir=runs
writer.add_graph(net.float(), batch_tb['image'])
#writer.close()

# Random Samples in a Projector ( PCA or T-SNE)
n_images = 200
max_range = len(img_list)
img_group = []
class_group = []
for i in range(n_images):
    sample_test = transformed_dataset.__getitem__(random.randint(0,max_range))
    img_group.append(sample_test['image'][0])
    class_group.append(sample_test['class_name'])

img_group_t = torch.stack(img_group)
class_group_t = torch.stack(class_group)

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# %% Projector - Might need Tensorboard Terminal Restart
features = img_group_t.view(-1, 32 * 32)
writer.add_embedding(features,
                    metadata=class_group_t,
                    label_img=img_group_t.unsqueeze(1))

writer.close()



