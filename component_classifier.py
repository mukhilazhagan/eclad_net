# %%
from __future__ import print_function, division
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from definitions import *
from PIL import Image
from classes import *
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
isValidation = True
ratio1 = 2
ratio2 = 2

# Store or not values from validation
storeValidationResults = False




# %% Load data
if is_read_data:
    img_list_test, class_list_test_t = read_images(train=False)
    img_list_train, class_list_train_t = read_images(train=True)
    

class_list_test = onehotencoding_class(class_list=class_list_test_t)
class_list_train = onehotencoding_class(class_list=class_list_train_t)

   
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
    net = GhostNet(ratio1,ratio2)
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
# Third row : train FINAL accuracy
# Forth row : train time
# Fifth row : validation FINAL accuracy
# Six row : validation time
# Seventh tow : nb trainable parameters
validation_results = [[]]
# Store all loss capture for each set of parameters
train_loss_capture = [[]]
# Store all accuracy capture for each set of parameters
train_accuracy_capture = [[]]
# Store all accuracy capture for each set of parameters
validation_accuracy_capture = [[]]

for ratio1 in range(2,6):

    for ratio2 in range(2,6):
        
        loss_capture = []
        accuracy_capture = []

        if isValidation:
            accuracy_capture_v = []
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

            #################################
            ## END TRAIN FOR CURRENT EPOCH ##
            #################################
            
            # Mean loss for the current epoch of all loss computed for each batch (batchsize : 4)
            running_loss = running_loss / len(trainloader)  
            loss_capture.append(running_loss)
            accuracy_capture.append(correct/total)
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            print("Accuracy : {} - Ratio : {}/{} \n".format(correct/total,correct,total))
            
            ## Validation for each epochs
            if isValidation:
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
                print(f'Validation Loss at epochs {epoch}:{validation_loss/len(validationloader)}')
                print(f'Correct Predictions in validation at epoch{epoch} : {vcorrect}/{vtotal}')
                accuracy_capture_v.append(vcorrect/vtotal)
            
        #########################
        ## END EPOCHS FOR LOOP ##
        #########################
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
        
        ## Retrieve data
        train_loss_capture.append(loss_capture)
        train_accuracy_capture.append(accuracy_capture)
        if isValidation:
            validation_accuracy_capture.append(accuracy_capture_v)
        ## Retrieve some usefull data for last epoch
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
        
        ## Nb weight and bias used by the model
        # Represent data needed to store and exploit the model
        pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        tmp_valid.append(pytorch_total_params_trainable)
        # Update count to store new informations of validation in validation results
        validation_results.append(tmp_valid)
    if not(isValidation):
        break


# %% Save model
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("nb parameters : {}".format(pytorch_total_params_trainable))

torch.save(net.state_dict(), "runs/model/currentNet.pt")
    
# %% STORE VALIDATION RESULTS
if storeValidationResults and isValidation:
    np.savetxt("validation/parameters.csv",validation_results[1:] , delimiter=",")  
    np.savetxt("validation/ValdiationAccuracies.csv",validation_accuracy_capture[1:] , delimiter=",")  
    np.savetxt("validation/TrainLosses.csv",train_loss_capture[1:] , delimiter=",")    
    np.savetxt("validation/TrainAccuracies.csv",train_accuracy_capture[1:] , delimiter=",")  
# %% TEST AND PLOT INTERESTING VALUES



if isValidation:
    print("WARNING, test phase not available in validation Mode")
    print("Make sure to previously define the right hyperparameter through validation")
    print("Then, unset \"isValidation\" to train the model with the right hyperparameters")
    quit()
    
    
test_loss = 0.0
correct, total = 0,0
i = 0
mse1 = 0.0
mse2 = 0.0
plot = False
test = True

tic = time.perf_counter()
for i_batch, sample_batched in enumerate(testloader):
    inputs = sample_batched['image']
    labels = sample_batched['class_name']
    i += 1
    if test == True:
        # if i%10 == 0:
        #     saliency(inputs,net)
        # else:   
        output, tmp_mse1, tmp_mse2 = net(inputs,test=test,plot=plot)   
        mse1 += tmp_mse1
        mse2 += tmp_mse2
    else:
        output = net(inputs,test=test)
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
print(f'MSE Mean Conv 1 : {mse1/i}')
print(f'MSE Mean Conv 2: {mse2/i}')


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
