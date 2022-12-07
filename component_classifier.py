# %%
from __future__ import print_function, division
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from definitions import *
from PIL import Image

import os
import torch
import torchvision
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import warnings
import time 
import math
import tensorflow as tf
import tensorboard as tb

warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

# Define the network type : Ghost or Usual
GhostType = True
is_read_data = True

# Validation of GhostParameters or not
isValidation = False
FristRatio = 3
SecondRatio = 3

# Store or not values from validation
storeValidationResults = False




# %% Load data
if is_read_data:
    img_list_test, class_list_test_t = read_images(train=False)
    img_list_train, class_list_train_t = read_images(train=True)
    

class_list_test = onehotencoding_class(class_list=class_list_test_t)
class_list_train = onehotencoding_class(class_list=class_list_train_t)

################## %% 
# %%Classes

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
        new_channels =  init_channels*(ratio-1)
    
        ## Primary standard convolution + MP + ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride),
            nn.ReLU(inplace=True)
        )
        
    
        ### Secondary depthwise convolution + MP + ReLU
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels),
        # groups allow to perform convolution only once for each matrix 
            nn.ReLU(inplace=True)
        )


    def forward(self, x, test=False):
        tic = time.perf_counter()
        x1 = self.primary_conv(x)
        toc = time.perf_counter()
        print(f"Classic Conv in {toc-tic:0.12f} seconds")        
        
        ## WARNINGS, VALUE 
        if test:
            plt.figure()
            # plot conv output from the first layer for the first figure of the batch
            for i in range(self.init_channels):
                plt.subplot(self.init_channels//2+1, 3, i+1)
                # print(torch.max(torch.abs(x_temp1[0,i,:,:])))
                plt.imshow(x1[0,i,:,:].detach().numpy(),cmap='gray', vmin=0, vmax= 1.5)
                plt.xticks([])
                plt.yticks([])
            plt.suptitle("Classic Convolutional Layer, MSE : {0:.4f}".format(MSE(x1)), fontsize=16)
                
        tic = time.perf_counter()
        x2 = self.cheap_operation(x1)
        toc = time.perf_counter()
        print(f"Cheap Conv in {toc-tic:0.12f} seconds")
        
                ## WARNINGS, VALUE 
        if test:
            plt.figure()
            # plot conv output from the first layer for the first figure of the batch
            for i in range(self.new_channels):
                plt.subplot(self.new_channels//2+1, 3, i+1)
                # print(torch.max(torch.abs(x_temp1[0,i,:,:])))
                plt.imshow(x2[0,i,:,:].detach().numpy(),cmap='gray', vmin=0, vmax= 1.5)
                plt.xticks([])
                plt.yticks([])
            plt.suptitle("Cheap Operation Results, MSE : {0:.4f}".format(MSE(x2)), fontsize=16)

        ### Stack Standard and Depthwise
        out = torch.cat([x1,x2], dim=1)
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
    def forward(self, x, test=False):
        # Conv + ReLu + Pool (First Layer)
        # tic = time.perf_counter()
        if test:
            x_temp1_ = self.conv1(x)
            # toc = time.perf_counter()
            # print(f"Trained in {toc-tic:0.4f} seconds")
            # Ghost Module
            # x = self.pool(self.conv1.forward(x))
            # plot conv output from the first layer for the first figure of the batch
            plt.figure()
            plt.subplot(5,3,2)
            plt.imshow(x[0].detach().numpy().transpose(1, 2, 0))
            for i in range(12):
                plt.subplot(5, 3, i+1+3)
                # print(torch.max(torch.abs(x_temp1[0,i,:,:])))
                plt.imshow(x_temp1_[0,i,:,:].detach().numpy(),cmap='gray', vmin=0, vmax= 1.5)
                plt.xticks([])
                plt.yticks([])
            plt.suptitle("First Convolutional Layer, MSE : {0:.4f}".format(MSE(x_temp1_)), fontsize=16)
            
            x_temp1 = self.pool(F.relu(x_temp1_))
            
        ## Computeall in once if no conv plot required
        else : 
            tic = time.perf_counter()
            xtemp1_ = F.relu(self.conv1(x))
            toc = time.perf_counter()
            print(f"Classix Net : Classic Conv 1 in {toc-tic:0.12f} seconds")
            x_temp1 = self.pool(xtemp1_)
        
        
        # Conv + ReLu + Pool (Second Layer)
        if test:
            x_temp2_ = self.conv2(x_temp1)
            # plot conv output from the second layer for the first figure of the batch
            plt.figure()
            plt.subplot(9,3,2)
            plt.imshow(x[0].detach().numpy().transpose(1, 2, 0))
            for i in range(24):
                plt.subplot(9, 3, i+1+3)
                plt.imshow(x_temp2_[0,i,:,:].detach().numpy(),cmap='gray', vmin=0, vmax =1.5)
                plt.xticks([])
                plt.yticks([])
            plt.suptitle("Second Convolutional Layer, MSE : {0:.4f}".format(MSE(x_temp2_)), fontsize=16)
            
            x_temp2 = self.pool2(F.relu(x_temp2_))
        else:
            
            tic = time.perf_counter()
            xtemp2_ = F.relu(self.conv2(x_temp1))
            toc = time.perf_counter()
            print(f"Classix Net : Classic Conv 2 in {toc-tic:0.12f} seconds")
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
    def forward(self, x, test=False):
        # Conv + ReLu + Pool (First Layer)
        # tic = time.perf_counter()
        x_temp1 = self.pool1(F.relu(self.ghost1(x,test)))
        # toc = time.perf_counter()
        # print(f"Trained in {toc-tic:0.4f} seconds")
        # Conv + ReLu + Pool (Second Layer)
        x_temp2 = self.pool2(F.relu(self.ghost2(x_temp1,test)))
        if test:
            plt.figure()
            plt.imshow(x[0].detach().numpy().transpose(1, 2, 0))
        # x = self.pool2(self.conv2.forward(x))
        # -1 re arrange array regarding the second parameter
        ## Error "shape '[-1, 400]' is invalid for input of size 4096"
        x_ = x_temp2.view(-1, 24*5*5)
        x_ = F.relu(self.fc1(x_))
        # No relu for fc2 cause we use softMax to end up with probability for the class
        x_ = self.fc2(x_)
        x_ = self.sm1(x_)
        
        return x_
        
###################
### END Classes ###
###################
   
# %% DATA FOMATTING
# transforms.Compose([Rescale(32), ToTensor()])) : applique d'abord une transformation Rescale puis toTensor
# En gros ça transforme nos images 30x30 en 32x32
BornValidation = int(math.ceil(len(img_list_train)*0.4))
a = img_list_train[len(img_list_train)-BornValidation:-1]
concImgTrain = np.concatenate((img_list_train[:BornValidation],img_list_train[len(img_list_train)-BornValidation:-1]), axis=0)
concLabelTrain = np.concatenate((class_list_train[:BornValidation],class_list_train[len(img_list_train)-BornValidation:-1]), axis=0)


transformed_dataset_train = Logo_Dataset(concImgTrain, concLabelTrain  , transform = transforms.Compose([Rescale(32), ToTensor()]))
trainloader = DataLoader(transformed_dataset_train, batch_size=4,
                        shuffle=True, num_workers=0)

concImgValidation = img_list_train[BornValidation:len(img_list_train)-BornValidation]
concLabelValidation = class_list_train[BornValidation:len(class_list_train)-BornValidation]

transformed_dataset_validation = Logo_Dataset(concImgValidation, concLabelValidation  , transform = transforms.Compose([Rescale(32), ToTensor()]))
validationloader = DataLoader(transformed_dataset_validation, batch_size=4,
                        shuffle=True, num_workers=0)

transformed_dataset_test = Logo_Dataset(img_list_test, class_list_test  , transform = transforms.Compose([Rescale(32), ToTensor()]))
testloader = DataLoader(transformed_dataset_test, batch_size=1,
                        shuffle=False, num_workers=0)

# %% HYPERPARAMETERS 

## WARNING !! 
# When validation is wanted, GhostNet is defined later in the program
# It's redefined for every new set of hyperparameters in the TRAINING PHASE

if GhostType == False:
    net = Net()
    # Usual criterion for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
elif not(isValidation):
    net = GhostNet(FristRatio,SecondRatio)
    # Usual criterion for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
else: 
    print("## WARNING !! When validation is wanted, GhostNet is defined later in the program \n It's redefined for every new set of hyperparameters in the TRAINING PHASE")




# %% TRAINING PHASE

# List Capturing Loss Curve
epochs = 10
plotLoss = False
# First row : ratio1 
# Second row : ratio2
# Third row : train final accuracy
# Forth row : train time
# Fifth row : validation accuracy
# Six row : validation time
validation_results = [[]]
# Store all loss capture for each set of parameters
validation_loss_capture = [[]]
# Store all accuracy capture for each set of parameters
validation_accuracy_capture = [[]]

for ratio1 in range(2,5):

    for ratio2 in range(2,5):
        
        loss_capture = []
        accuracy_capture = []
        if isValidation:
            # Define the tested Model
            net = GhostNet(ratio1,ratio2)
            # Usual criterion for classification
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters())
            tmp_valid = []
            tmp_valid.append(ratio1)
            tmp_valid.append(ratio2)
            
            
        tic = time.perf_counter()
        for epoch in range(epochs):  # loop over the dataset multiple times
            
            # Use to compute mean loss every 20 batch. Not concretly used by model in training 
            running_loss = 0.0

            # Used to check accuracy of the model in training 
            # Should be used only for the developpement phase
            total = 0
            correct = 0
            
            # Pass through the network with 4 differents images before backwarding
            for i_batch, sample_batched in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = sample_batched['image']
                labels = sample_batched['class_name']
                
                # zero the parameter gradients
                # justif : https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                # En gros, sert à pas utiliser les descentes de gradient déjà exploitées dans les étapes précédentes. 
                optimizer.zero_grad()
                
                # forward + backward + optimize
                # Check ghosttype because validation only for ghost type
                if GhostType and isValidation:
                    outputs = net(inputs)
                else:
                    outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                
                # Used to check accuracy of the model
                for o,l in zip(torch.argmax(outputs,axis = 1),labels):
                    if l==o:
                        correct += 1
                    total += 1

            # Mean loss for the current epoch of all loss computed for each batch (batchsize : 4)
            running_loss = running_loss / len(trainloader)  
            loss_capture.append(running_loss)
            accuracy_capture.append(correct/total)
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            print("Accuracy : {} - Ratio : {}/{} \n".format(correct/total,correct,total))
            
        toc = time.perf_counter()
        print('Finished Training')
        if plotLoss:
            plt.figure(1)
            plt.subplot(211)
            plt.plot(loss_capture)
            plt.title("Loss Curve")
            plt.subplot(212)
            plt.plot(accuracy_capture)
            plt.title("Training accuracy")
            plt.show()
        print(f"Trained in {toc-tic:0.4f} seconds")  
        # Break is not in validation mode to only train once and not perform validation test
        if not(isValidation):
            break
        
        
        validation_loss_capture.append(loss_capture)
        validation_accuracy_capture.append(accuracy_capture)
        tmp_valid.append(correct/total)
        tmp_valid.append(toc-tic)
        i = 0
        validation_loss = 0.0
        vcorrect, vtotal = 0,0
        
        vtic = time.perf_counter()
        for vi_batch, vsample_batched in enumerate(validationloader):
            vinputs = vsample_batched['image']
            vlabels = vsample_batched['class_name']
            i += 1
            voutput = net(vinputs,test=False)
            #print(torch.argmax(output,axis = 1))
            for o,l in zip(torch.argmax(voutput,axis = 1),vlabels):
                if o == l:
                    vcorrect += 1
                vtotal += 1
            vloss= criterion(voutput,vlabels)
            validation_loss += vloss.item() * vinputs.size(0)
        vtoc = time.perf_counter()
        print(f'Validation Loss:{validation_loss/len(validationloader)}')
        print(f'Correct Predictions in validation : {vcorrect}/{vtotal}')
        tmp_valid.append(vcorrect/vtotal)
        tmp_valid.append(vtoc-vtic)
        # Update count to store new informations of validation in validation results
        validation_results.append(tmp_valid)
    if not(isValidation):
        break


# %% 


print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    
# %% STORE VALIDATION RESULTS
if storeValidationResults and isValidation:
    np.savetxt("validation/parameters.csv",validation_results[1:] , delimiter=",")  
    np.savetxt("validation/losses.csv",validation_loss_capture[1:] , delimiter=",")    
    np.savetxt("validation/accuracies.csv",validation_accuracy_capture[1:] , delimiter=",")  
# %% TEST AND PLOT INTERESTING VALUES



if isValidation:
    print("WARNING, test phase not available in validation Mode")
    print("Make sure to previously define the right hyperparameter through validation")
    print("Then, unset \"isValidation\" to train the model with the right hyperparameters")
    quit()
    
    
test_loss = 0.0
correct, total = 0,0
i = 0
plot = False


tic = time.perf_counter()
for i_batch, sample_batched in enumerate(testloader):
    inputs = sample_batched['image']
    labels = sample_batched['class_name']
    i += 1
    if plot == True:
        # if i%10 == 0:
        #     saliency(inputs,net)
        # else:   
        output = net(inputs,test=True)   
    else:
        output = net(inputs,test=False)
    #print(torch.argmax(output,axis = 1))
    for o,l in zip(torch.argmax(output,axis = 1),labels):
        if o == l:
            correct += 1
        total += 1
    loss = criterion(output,labels)
    test_loss += loss.item() * inputs.size(0)
    
toc = time.perf_counter()
print(f"Tested in {toc-tic:0.4f} seconds")
print(f'Testing Loss:{test_loss/len(testloader)}')
print(f'Correct Predictions: {correct}/{total}')



# # %%
# # Tensorboard
 
# # default `log_dir` is "runs" 
# writer = SummaryWriter('runs/res_and_cap_exp')
# dataiter = iter(trainloader)
# batch_tb = next(dataiter)
# img_grid = torchvision.utils.make_grid(batch_tb['image'])

# # %%
# # write to tensorboard
# writer.add_image('four images', img_grid)
# ## At this stage Run in command Line tensorboard --logdir=runs
# writer.add_graph(net.float(), batch_tb['image'])
# #writer.close()

# # Random Samples in a Projector ( PCA or T-SNE)
# n_images = 200
# max_range = len(img_list_test)
# img_group = []
# class_group = []


# for i in range(n_images):
#     # sample_test is a list {image,class}
#     sample_test = transformed_dataset_train.__getitem__(random.randint(0,max_range))
#     img_group.append(sample_test['image'][0])
#     class_group.append(sample_test['class_name'])

# # Stack output images (actually feature vector) and corresponding class in tensor
# img_group_t = torch.stack(img_group)
# class_group_t = torch.stack(class_group)

# # %%
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# # %% Projector - Might need Tensorboard Terminal Restart
# features = img_group_t.view(-1, 32 * 32)
# writer.add_embedding(features,
#                     metadata=class_group_t,
#                    label_img=img_group_t.unsqueeze(1))

# writer.close()
# while True:
#     print('end')


# # %%
