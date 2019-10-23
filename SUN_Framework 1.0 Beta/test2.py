##test the synapse plasticity


from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import Neuron



###########################################################################
##   Spike-based input test
###########################################################################




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
n_syn = 125
# Spiking frequency in Hz
f = 45
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
neuron=Neuron.Neuron(max_spikes=T,n_syn=n_syn,u_rest=-65,u_thresh=-50,tau=1.7,R=1.7)
for time in range(T):
    i,u = neuron.forward(syn_has_spiked)
    I.append(i)
    U.append(u)

#print(neuron.spike_train)
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
plt.figure()
plt.plot([spike for spike in neuron.spike_train])
plt.title('Spike response')
plt.xlabel('Step(1000ms)')

plt.show()







I = []
U = []
neuron.time=0
for time in range(T):
    i,u = neuron.forward(syn_has_spiked)
    I.append(i)
    U.append(u)

#print(neuron.spike_train)
###########################################################################
##   Visualizing
###########################################################################
plt.rcParams["figure.figsize"] = (12, 6)


# Draw the input current and the membrane potential

plt.figure()
plt.plot([u for u in U])
plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')
plt.figure()
plt.plot([spike for spike in neuron.spike_train])
plt.title('Spike response')
plt.xlabel('Step(1000ms)')

plt.show()



'''

###########################################################################
##   Current-based input test
###########################################################################



###########################################################################
##   Give the useful parameters
###########################################################################
# Duration of the simulation in ms
T = 3000
# Duration of each time step in ms
dt = 1
# Number of iterations = T/dt
steps = int(T / dt)
# Number of synapses
n_syn = 2
###########################################################################
##   We offer the spike trains as the normal distribution
###########################################################################
mu = 8
sigma = 1.75
np.random.seed(0)
I = np.random.normal(mu, sigma,steps)
#I=np.ones(steps)*7.5
noise=np.random.normal(0,1.4,steps)
I=I+noise

I=torch.tensor(I)
U=[]
###########################################################################
##   As the input into our neuron .
###########################################################################
neuron=Neuron.Neuron_input_I(max_spikes=steps,n_syn=n_syn,dt=dt)
for time in range(steps):
    u = neuron.forward(I)
    U.append(u)

#print(neuron.spike_train)

###########################################################################
##   Visualizing
###########################################################################
plt.rcParams["figure.figsize"] = (12, 6)
# Draw the input current and the membrane potential
plt.figure()
plt.plot([i for i in I])
plt.title('Synaptic input')
plt.ylabel('Input current (I)')
plt.xlabel('Time (msec)')

plt.figure()
plt.plot([u for u in U])
#plt.axhline(y=1.0, color='r', linestyle='-')
plt.title('LIF response')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Step(1000ms)')

plt.figure()
plt.plot([spike for spike in neuron.spike_train])
plt.title('Spike response')
plt.xlabel('Step(1000ms)')
plt.text(1000, 0.8, "Spike rate = "+str(neuron.spike_count/T*1000), size = 15,
         alpha = 0.8,color = "r",weight = "bold",bbox = dict(facecolor = "y", alpha = 0.8),
         family = "fantasy")
plt.show()



#TODO:然后需要检验一下rC和rR是否收敛


'''

