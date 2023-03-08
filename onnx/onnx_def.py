import time
import numpy as np

def testModelONNX(model,testset,labels_onehot,device):
    acc = 0
    correct, total = 0.0, 0.0
    inf_time = 0
    outputs = []

    tic = time.perf_counter()
    for img in testset:
        outputs.append(model.run(None, {'input': np.reshape(img, (1, 3, 32, 32).to(device)   )})[0][0])
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