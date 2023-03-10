# Weights-Pruning

## Pruning of ResNet50 Network with Cifar10 Dataset

![boxing](https://github.com/MichaelB0312/Weights-Pruning/blob/main/images/prune_vis.gif)
## By: [Naomi Shapiro](https://github.com/naomishmish)   and   [Michael Berko](https://github.com/MichaelB0312)



### In this project, we will explore how reducing weights in a given model affects the accuracy of the model, in different words exploring the tradeoff between low memory and high accuracy. 

## Agenda

* [Background](#background)
* [Dataset](#dataset)
* [Model](#model)
* [Pruning Process](#pruning-process)
* [Parameters](#prerequisites)
* [Running Instructions](#running-instructions)
* [Results](#Results)
* [Prerequisites](#prerequisites)
* [Files in the repository](#files-in-the-repository)
* [References](#references)

## Background
In this project, we will explore how reducing weights in a given model affects the accuracy of the model, in different words exploring the tradeoff between low memory and high accuracy. 
We will gradually remove weights with the lowest L1-norm in a trained model and see the test results. **Our assumption is that the lowest l1-norm weights are the least effective on model classification quality**. Our challenge will be how to decrease amount of weights without hurting the accuracy too much. In the end, we will determine what is the optimal percent of weights that can be removed with keeping high accuracy.
We took the idea for this project from:[Pruning Algorithm to Accelerate Convolutional Neural Networks for Edge Applications](http://arxiv.org/abs/2005.04275), by J. Liu, S. Tripathi, U. Kurup, and M. Shah.

## Dataset
We used the [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset in this project.
Cifar10 is the subset labeled dataset collected from 80 million tiny images dataset and contains 10 classes.

## Model
We used the [ResNet50](http://image-net.org/challenges/LSVRC/2015/) Model in this project.
Deep Convolutional neural networks are great at identifying features from images, and adding more layers generally provides better accuracy. However, adding more layers to a suitable deep model just increases the training error and does not give better results.
The problem is the vanishing gradients, i.e. the gradients decrease in the first layers as the network becomes more deeper.
The ResNet50 network solves this problem by creating shortcut connections that simply perform identity mappings. This allows the running tasks to earn depth benefits while reasonably maintaining (reducing) the computational expense.
![boxing](https://github.com/MichaelB0312/Weights-Pruning/blob/main/images/resnet50.png)

## Pruning Process
Pruning generally means cutting down parts of the network that contribute less or nothing to the network during inference. This results in models that are smaller in size, more memory-efficient, more power-efficient, and faster at inference with minimal loss in accuracy.
In this project, we will use connection pruning, particularly L1 norm pruning, which removes a specified number of weights units with the lowest L1 norm.

![boxing](https://github.com/MichaelB0312/Weights-Pruning/blob/main/images/pruning_process.png)

## Running Instructions
#### Stage 1: Run `main.py`
It's important to mention that Pruning process occures on **inference time**. Before that you should make the training of our model in `main.py`.
Basically, you can run it directly by your favoutire IDE.
Naturally, you would probably be inquisitive about the relations between hyprer-parameters and Pruning performance.
Thus, **we're offering interactive I/O for hyprer-parameters tuning with `argparse`:**

|Parameter | Type | Input Command | Recommended Value | Description| 
|-------|------|--------------------------------------------|----|--------------|
|batch_size| int | ```python run.py --batch_size <your value>``` | 128 | mini-batch size |
|learning_rate| float | ```python run.py --learning_rate <your value>```|0.01| initial optimizer's learning rate |
|momentum| float | ```python run.py --learning_rate <your value>``` |0.9| Optionally for Adam's optimizer |
|weight_decay| float | ```python run.py --weight_decay <your value>``` | 5e-4 | regularization parameter | 
|epochs| int | ```python run.py --epochs <your value>``` | 60 | Amount of running on all data| 
|T_max| int | ```python run.py --T_max <your value>``` | 20 |  [Cosine Annealing parameter](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) |

Recommended values were selected empirically as the best parameters for pruning process.
At the end of training, you should notice that you get locally the file: `./checkpoints/cifar10_resnet50_ckpt_epoch60.pth` which concludes the **checkpoints** of our model. 

#### Stage 2: Run `pruning.py`
As you see from our last remark from Stage 1, you should first ensure that you have checkoints file because our Pruning Process occures on post-training.
We are using the package `torch.nn.util.prune` and make the following process:
```python
for percent in prune_percents:
    # load the trained model
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
```
As you cans see, we examine different percents of amount of weights to be omitted from the pretrained model. In the internal loop we use `prune.l1_unstructured` which cuts for each layer the smallest L1-norm weights. We make the seperation for conv. layers and linear layers because we've found that pruning the final linear layers affect the accuracy, especially the last decisive-softmax layer.
**Where is the process of removing weights?** well, we have a mask which is an internal state buffer to stay only part of the weights and can be obtained by `model.named_buffers()`. Officialy we have only the parameter of the original weights named `weight_orig` (obtained by `model.named_parameters()`). This parameter are multiplied by the mask and the result is stored in another pruning's attribute called `model.weight`.This multplication is effectively, the pruning. It occures implictly by a callback invoked before each forward-pass by Pytorch's `forward_pre_hooks`.

# Results
After having a baseline, we started to cut weights from the model. In each iteration, we removed a higher percentage of weights, from 0% in the first iteration to 100% in the last one, with a step size of 5%. In the end, we got a graph of the accuracy as a function of the pruning percent. We chose hyper-paramters and optimizer as decribed above.

We've got the following result:

![boxing](https://github.com/MichaelB0312/Weights-Pruning/blob/main/images/prune_results.png)

As we can see, it's possible to removing 40% of the weights, **massive saving of resources and memory** while having **minimal damage on our high accuracy!** 



# Prerequisites
|Library         | Version |
|--------------------|----|
|`Python`|  `3.9 (Anaconda)`|
|`numpy`| `1.22.3`|
|`torch`| `1.11.0`|
|`torchvision`| `0.12.0`|
|`matplotlib`| `3.5.2 `|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`run.py`| where we train our model|
|`Build_Resnet.py`| consists the elemntary blocks for resnet and their integration to the whole architecture|
|`pruning.py`| post-training pruning procees as described above|
|images| Readme.md images|



## References
* [Pruning Algorithm to Accelerate Convolutional Neural Networks for Edge Applications](http://arxiv.org/abs/2005.04275), by J. Liu, S. Tripathi, U. Kurup, and M. Shah.
* https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py


