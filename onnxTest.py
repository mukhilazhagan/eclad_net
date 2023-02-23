import onnx
import argparse
import onnxruntime
import definitions
import numpy as np
import torchvision.transforms as transforms
import torch
import os

##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)
args = parser.parse_args()

isGhostNet = args.ghost

onnx_path = "runs/model/onnx"
# Load the ONNX model
onnx_file = os.path.join(onnx_path,"ghostEclad_{}_{}.onnx".format(args.ratio1,args.ratio2)) if isGhostNet else os.path.join(onnx_path,"ecladNet.onnx")

model = onnx.load(onnx_file)
# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))




## RUNTIME
# Define model 
ort_session = onnxruntime.InferenceSession(onnx_file)

# Retrieve input 
img_list_test, class_list_test_t = definitions.read_images(train=False)

class_list_test_t = definitions.onehotencoding_class(class_list_test_t)

resize_transform = transforms.Resize((32, 32))


input_data = np.array(img_list_test)  # Convert the list of images to a numpy array
input_data = np.transpose(input_data, (0, 3, 1, 2))  # Change the layout from NHWC to NCHW
input_data = input_data.astype(np.float32)  # Convert the data type to float32
input_data = torch.tensor(input_data)
input_data = torch.stack([resize_transform(img) for img in input_data])
input_data = np.array(input_data)



acc,tim_inf = definitions.testModelONNX(ort_session, input_data, class_list_test_t)
