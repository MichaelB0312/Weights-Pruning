import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_scheduler
from Build_Resnet import Resnet50Model
from run import calculate_accuracy
from run import model
from run import testloader

# device - cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pruning process
prune_percents = np.linspace(0, 1, 21)  # range of pruning
test_acc = []

for percent in prune_percents:
    # load the trained model
    #model = Resnet50Model().to(device)
    state = torch.load(f'./checkpoints/cifar10_resnet50_ckpt_epoch60.pth', map_location=device)
    model.load_state_dict(state['net'])

    # performing the pruning
    for name, module in model.named_modules():
        if percent == 0:
            continue
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module=module, name='weight', amount=percent)
        if isinstance(module, torch.nn.Linear) and name != 'output':
            prune.l1_unstructured(module=module, name='weight', amount=percent)

    print(f'PRUNED {percent}')

    # running test after pruning
    test_accuracy, confusion_matrix = calculate_accuracy(model, testloader, device)
    test_acc.append(test_accuracy)
    print("test accuracy: {:.3f}%".format(test_accuracy))


# plot pruning graph
plt.figure()
plt.plot(prune_percents, test_acc, linewidth=3.0, marker='o')
plt.title('Resnet50 with l1 pruning', fontweight="bold", size=20)
plt.xlabel('prune percent [%]', fontsize = 18)
plt.ylabel('accuracy', fontsize = 18)
plt.show()
