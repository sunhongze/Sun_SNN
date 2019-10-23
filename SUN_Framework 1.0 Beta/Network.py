
from __future__ import print_function
import matplotlib.pyplot as plt
from Neuron import Neuron
import torch
import numpy as np

'''
class Sun_FC_Network(Neuron):
    def __init__(self,R=1.0, u_rest=-0.1, tau=10.0, dt=1,
                 u_thresh=1.0, T_rest=4.0,
                 n_syn=10, max_spikes=50, q=1.5, tau_syn=10.0,
                 shape=[2,5,1],
                 ):
        super(Sun_FC_Network,self).__init__(R=R, u_rest=u_rest, tau=tau, dt=dt,
                 u_thresh=u_thresh, T_rest=T_rest,
                 n_syn=n_syn, max_spikes=max_spikes, q=q, tau_syn=tau_syn)

        ## Create the Network , NOTE , the input layer is not included
        self.shape=shape
        self.hidden_layer=[]
        for i in range(len(shape)-2):
            self.hidden_layer.append([Neuron(n_syn=self.shape[i]) for t in range(self.shape[i+1])])
        ############################################################

    def train(self,input_spike_trains):

        spike_trains=input_spike_trains
        spike_trains_new=[]
        for layer in self.hidden_layer:
            for neuron in layer:
                for time in range(self.max_spikes):
                    neuron.forward(spike_trains)

                spike_trains_new.append(neuron.spike_train)
            spike_trains=torch.tensor(spike_trains_new)


        return True


'''
class Sun_FC_Network(object):
    def __init__(self,n_syn=10, max_spikes=50, shape=[2,5,1]):

        ## Create the Network , NOTE , the input layer is not included
        self.shape=shape
        self.hidden_layer=[]
        self.output_layer=[]
        self.max_spikes=max_spikes
        # We make the hidden layer to be a group layer by layer
        for i in range(len(shape)-2):
            self.hidden_layer.append([Neuron(n_syn=self.shape[i],max_spikes=self.max_spikes) for t in range(self.shape[i+1])])
        for i in range(self.shape[-1]):
            self.output_layer.append(Neuron(n_syn=self.shape[-2],max_spikes=self.max_spikes))
        ############################################################

    def train(self,input_spike_trains):

        # spike trains is a set for each neuron's spike train in the same layer
        # And of course , the output's spike train is the result of our Network.
        # output_U is a set for each neuron's U train in the output layer
        spike_trains=input_spike_trains
        spike_trains_new=[]
        output_U=[[] for i in range(self.shape[-1])]
        output_I = [[] for i in range(self.shape[-1])]
        ###########################################################################
        #   We forward the spike trains through the network's hidden layer
        ###########################################################################
        for layer in self.hidden_layer:
            for neuron in layer:
                for time in range(self.max_spikes):
                    neuron.forward(spike_trains)
                spike_trains_new.append(neuron.spike_train.numpy())
            spike_trains=torch.tensor(spike_trains_new)
            spike_trains=torch.t(spike_trains)
            spike_trains_new=[]

        ###########################################################################
        #   We forward the spike trains through the network's hidden layer
        ###########################################################################
        count=0
        for neuron in self.output_layer:
            for time in range(self.max_spikes):
                neuron.forward(spike_trains)
                output_U[count].append(neuron.U)
                output_I[count].append(neuron.I_input)
            count+=1
            spike_trains_new.append(neuron.spike_train.numpy())
        spike_trains=torch.tensor(spike_trains_new)
        spike_trains=torch.t(spike_trains)

        return spike_trains,output_U,output_I



###########################################################################
##   Give the useful parameters
###########################################################################
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Number of input synapses  This is used to genetate the input spike trains
n_syn = 30
# Spiking frequency in Hz
f = 35
# We need to keep track of input spikes over time
syn_has_spiked = np.full((steps, n_syn), False)

###########################################################################
##   We offer the spike trains as the poisson distribution
###########################################################################
for step in range(steps):
    r = np.random.uniform(0, 1, size=(n_syn))
    syn_has_spiked[step, :] = r < f * dt * 1e-3
syn_has_spiked=torch.tensor(syn_has_spiked)

# We define the synaptic efficacy as a random vector
W = np.random.normal(1.0, 0.5, size=n_syn)
# Output variables
I = []
U = []

###########################################################################
##   We define the class Sun_Network
###########################################################################
Network=Sun_FC_Network(shape=[n_syn,50,20,1],max_spikes=T)
spike_trains,output_U,output_I=Network.train(syn_has_spiked)



###########################################################################
##   Visualing the result
###########################################################################
plt.rcParams["figure.figsize"] = (12, 6)
# Draw spikes
spikes = np.argwhere(syn_has_spiked)
spikes=spikes.numpy()
spiks = spikes.T
t=spikes[0]
s=spikes[1]
plt.figure()
plt.axis([0, T, 0, n_syn])
plt.title('Synaptic spikes')
plt.ylabel('spikes')
plt.xlabel('Time (msec)')
plt.scatter(t, s)

# Draw the input current and the membrane potential
plt.figure()
plt.plot([i for i in output_I[0]])
plt.title('Synaptic input')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([i for i in output_U[0]])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.show()