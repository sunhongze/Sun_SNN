
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn

class Neuron:
    def __init__(self,layer_shape,threshold,time=15,weight_mean=0.8,weight_std=0.02):
        self.layer=layer_shape
        self.threshold=threshold
        self.weight=self.init_weight(weight_mean,weight_std)
        self.neuron_potential=self.init_neuron_potential(self.layer)
        self.time=time
        self.neuron_spike=self.init_neuron_spike()
    def init_weight(self,weight_mean, weight_std):
        weight=[i for i in range(len(self.layer)-1)]
        for i in range(len(weight)):
            weight[i]=Parameter(torch.Tensor(self.layer[i+1], self.layer[i])).requires_grad_(False).normal_(weight_mean, weight_std)
        return weight

    def init_neuron(self,layer_shape):
        neuron_potential=[i for i in range(len(self.layer)-1)]
        return neuron_potential
    #TODO:##CONTINUE
    def init_neuron_spike(self,):
        return True
    def training(self,data_in):
        print(0)
        for time in range(self.time):
            for layer in range(len(self.layer)-1):
                self.neuron_potential[layer]=torch.sum(self.weight[layer]*data_in[time],dim=1)




                print(True)



        return True































