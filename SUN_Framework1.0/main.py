


########################################################################
##             Dataloader and Code
########################################################################

import Code
import numpy as np
import torch
Times=15
'''
kernels = [Code.DoGKernel(3, 3 / 9, 6 / 9),
		   Code.DoGKernel(3, 6 / 9, 3 / 9),
		   Code.DoGKernel(3, 7 / 9, 14 / 9),
		   Code.DoGKernel(3, 14 / 9, 7 / 9),
		   Code.DoGKernel(3, 13 / 9, 26 / 9),
		   Code.DoGKernel(3, 26 / 9, 13 / 9)]
'''
kernels = [Code.DoGKernel(3, 26 / 9, 13 / 9)]
filter = Code.Filter(kernels, padding = 1, thresholds = 50)
s1c1 = Code.S1C1Transform(filter,timesteps=Times)
path='/home/sunhongze/PycharmProjects/Wide_Narrow/data/MNIST_BIN'
trainset=Code.dataloader(path,s1c1,batch_size=1, shuttle=True)



########################################################################
##                      END
########################################################################

########################################################################
##                 Define the Network
########################################################################
from Network import Neuron
import torch.nn as nn
sun=Neuron([256,400,100,2],[20,20],time=Times)
for data,targets in trainset:
    data=torch.squeeze(data.view(1,Times,1,1,256)).float()
    sun.training(data_in=data)
    print(targets)



print('END')














