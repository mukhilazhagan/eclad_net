import torch
import torchvision
from classes import GhostNet,Net,Logo_Dataset,DataLoader,Rescale,ToTensor
from definitions import *


isGhostNet = True
if isGhostNet:
    netType = "GhostNet"
    model = GhostNet()
else:
    netType="ClassicNet"
    model = Net()

model.load_state_dict(torch.load("runs/model/currentNet.pt"))
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 32, 32)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("runs/model/traced_{}.pt".format(netType))