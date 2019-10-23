
from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import Neuron

###########################################################################
##   Give the useful parameters
###########################################################################
# Duration of the simulation in ms
T = 200
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Number of synapses
n_syn = 25
# Spiking frequency in Hz
f = 20
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
##   As the input into our neuron .
###########################################################################
neuron=Neuron.Neuron(max_spikes=T,n_syn=n_syn)
for time in range(T):
    i,u = neuron.forward(syn_has_spiked)
    I.append(i)
    U.append(u)

print(neuron.spike_train)
###########################################################################
##   Visualizing
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
plt.plot([i for i in I])
plt.title('Synaptic input')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([u for u in U])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.show()










