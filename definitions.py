# %%

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch 
from torchvision import transforms
import time


# %%
def read_images(train=False):
    if train:
        src = './dataset/train'
    else:
        src = './dataset/test'
    img_list =[]
    class_list = []

    for (root,dirs,files) in os.walk(src):
        print("\n Found "+str(len(dirs))+ " Classes\n")
        for i in range(len(dirs)):
            print("\n Processing Class: "+ str(dirs[i]))
            dir_str = src+'/'+dirs[i]+'/'

            for (root_1, dirs_1, files_1) in os.walk(dir_str):
                for j in range(len(files_1)):
                #for j in range(100): #Debug
                    temp_img_str = dir_str+files_1[j]
                    temp_img = plt.imread(temp_img_str)
                    #print(temp_img.shape)

                    # Only if images is Grayscale
                    if len(temp_img.shape)<3:
                        temp_conv = np.zeros((temp_img.shape[0],temp_img.shape[1],3))
                        temp_conv[:,:,0] = temp_img 
                        temp_conv[:,:,1] = temp_img
                        temp_conv[:,:,2] = temp_img
                        temp_img = temp_conv  
                    #temp_list = temp_list.append(temp_img)
                    temp_img = temp_img
                    img_list.append(temp_img[:,:,0:3]) # png to 3 channels
                    # Use to know number of images loaded
                    class_list.append(dirs[i])
        break
    
    return img_list, class_list

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

def test_data_load(img_list, class_list, idx):
    plt.imshow(img_list[idx])
    print("Image Size: ", img_list[idx].shape)
    print(class_list[idx])
    #print(img_list[idx])
    
def onehotencoding_class(class_list):
    # %% Converting classes to onehotencoding
    class_set = set(class_list)
    # Converte classes names to integer 
    class_int = [x for x in range(0,len(class_set))]

    class_one_hot = [list(class_set).index(class_list[x]) for x in range(len(class_list))]
    one_hot = torch.nn.functional.one_hot(torch.tensor(class_one_hot))
    #class_one_hot = [x for item in class_list] 

    # Problems where CrossEntropy Loss doesn't accept onehot value so reverting to class indices values
    #class_list = np.asarray(class_one_hot).astype(np.float32)
    class_list_oh = np.asarray(class_one_hot)

    # %% Check types
    
    return class_list_oh

def saliency(img, model):
    #inverse transform to get normalize image back to original form for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    #we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    #transoform input PIL image to torch.Tensor and normalize
    # input = img.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input 
    img.requires_grad = True
    #forward pass to calculate predictions
    preds = model(img)
    score, indices = torch.max(preds, 1)
    # #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    #apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(img[0])   
    #plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].detach().numpy().transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def MSE(convs, idx_images=0):
    # Function to define similarity between convolutional results
    # x store all the convolution results
    # dim of x : [a,b,c,d] with :
    #   a : idx of the images that's has been computed. idx_images is used to choose the image
    #       for which convolutions will be compared
    #   b : nb convolution. First layer has 12. Second Layer has 16
    #   c and d : dim of conv matrix. As it its computed after Pooling, might 
    #             be 14 for the first layer and 5 for the second layer
    
    
    summation = 0 
    # Finding total number of items in list
    nb_conv = convs.size(dim=1)
    count = 0
    for cur_im in range(0,convs.size(dim=0)):
        #looping through each convs of the list
        for i in range(0,nb_conv): 
            # loop again through each convs except the current one to compute MSE between all conv
            for j in range(i+1,nb_conv):
                for x in range(0, convs.size(dim=2)):
                    for y in range(0,convs.size(dim=3)):
                        count += 1
                        difference = (abs(convs[cur_im,i,x,y].item()) - abs(convs[cur_im,j,x,y].item()))**2  #finding the difference between observed and predicted value
                        squared_difference = difference**2  #taking square of the differene 
                        summation += squared_difference  #taking a sum of all the differences
    MSE = summation/count #dividing summation by total values to obtain average
    return MSE
    
def testModelONNX(model,testset,labels_onehot):
    acc = 0
    correct, total = 0.0, 0.0
    inf_time = 0
    outputs = []

    tic = time.perf_counter()
    for img in testset:
        outputs.append(model.run(None, {'input': np.reshape(img, (1, 3, 32, 32))})[0][0])
    toc = time.perf_counter()
    inf_time = toc-tic
    print(f"Tested all test set in {inf_time:0.4f} seconds\n")

    for o,l in zip(np.argmax(outputs,axis = 1),labels_onehot):
        if o == l:
            correct += 1
        total +=1
    acc = correct/total

    print("Accuracy : {}  ({}/{})".format(acc,correct,total))
    return acc,inf_time



def testModelPyTorch(model,testset,labels_onehot):
    acc = 0
    correct, total = 0.0, 0.0
    inf_time = 0
    outputs = []
    with torch.no_grad():
        tic = time.perf_counter()
        for img in testset:
            # Test false as no mse needed
            outputs.append(model(torch.unsqueeze(torch.from_numpy(img),0)))
        toc = time.perf_counter()
        inf_time = toc-tic
    print(f"Tested all test set in {toc-tic:0.4f} seconds\n")
    for cur_tens,l in zip(outputs,labels_onehot):
        o = torch.argmax(cur_tens,axis=1)
        if o.item() == l:
            correct += 1
        total +=1
    acc = correct/total
    print("Accuracy : {}  ({}/{})".format(acc,correct,total))
    return acc,inf_time