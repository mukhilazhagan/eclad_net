# %%
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
def read_images():
    src = './dataset'

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
