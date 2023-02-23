import torch
import argparse
from classes import GhostNet,Net
from definitions import *





##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)


args = parser.parse_args()

isGhostNet = args.ghost

# An example input you would normally provide to your model's forward() method.
dummy_input = torch.rand(1, 3, 32, 32)

onnx_path = "runs/model/onnx"
if not(os.path.exists(onnx_path)):
    os.makedirs(onnx_path)

if isGhostNet:
    netType = "GhostNet"
    model = GhostNet(args.ratio1,args.ratio2)
    model.load_state_dict(torch.load('runs/model/ghostNet_{}_{}.pt'.format(args.ratio1,args.ratio2)))
    model.eval()
    torch_out = model(dummy_input)
    print(torch_out)
    # Export to ONNX
    torch.onnx.export(model, dummy_input, os.path.join(onnx_path,"ghostEclad_{}_{}.onnx".format(args.ratio1,args.ratio2)),
                       export_params=True,
                         input_names = ['input'])
    

else:
    netType="ClassicNet"
    model = Net()
    model.load_state_dict(torch.load('runs/model/ecladNet.pt'))
    model.eval()
    torch_out = model(dummy_input)
    # Export to ONNX
    torch.onnx.export(model, dummy_input, os.path.join(onnx_path,"ecladNet.onnx"),
                    export_params=True,
                    input_names = ['input'])







