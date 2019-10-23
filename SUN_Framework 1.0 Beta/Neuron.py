
from __future__ import print_function
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt




class Neuron(object):
    def __init__(self, R=1.0, u_rest=-0.1, tau=10.0, dt=1,
                 u_thresh=1.0, T_rest=4.0,
                 n_syn=10, max_spikes=50, q=1.5, tau_syn=10.0
                 ):
        ###########################################################################
        #   During the intergrate : U'(t) = (R*I(t）-U(t)+Urest(t))/tau*dt + U(t)
        ###########################################################################
        # Membrane resistance in Ohm
        self.R = R
        # Membrane resting potential in mV
        self.u_rest = u_rest
        # Membrane time constant in ms
        self.tau = tau
        #Step in ms
        self.dt=dt

        ###########################################################################
        #   During the time at fire : We need to define a duration of resting time
        #   after refractory : T_rest .     Obviously , the threshold of U to fire
        #   is also needed .
        ###########################################################################
        # Membrane threshold potential in mV
        self.u_thresh = u_thresh
        # Duration of the resting period in ms
        self.T_rest = T_rest

        ###########################################################################
        #   Our neuron is multi_input to one output , so we need to define the number
        #   of the input_synapse. What's more , according to the formula :
        #   I(t) = ∑i(wi*∑f(Isyn(t-tif)))
        #   Isyn(t) = q/tau_syn*exp(-t/tau_syn
        ###########################################################################
        # Number of synapses
        self.n_syn = n_syn
        # Maximum number of spikes we remember
        self.max_spikes = max_spikes
        # The neuron synaptic 'charge'
        self.q = q
        # The synaptic time constant (ms)
        self.tau_syn = tau_syn
        # The synaptic efficacy
        self.w = torch.FloatTensor(np.random.normal(1.0, 0.5, size=self.n_syn))

        ###########################################################################
        #   Other variable that we need to define
        ###########################################################################
        self.U=self.u_rest
        self.spike_train=torch.tensor(np.full(self.max_spikes,False))
        self.spike=False
        self.time=0


    ###########################################################################
    #  Usually , we get the spike trains as the input . But our neuron is
    #  current_friendly . So we should transfer the spike trains
    #  to current(I) that we can deal with it by our neuron .
    ###########################################################################
    def forward(self,input_spike_trains):
        self.I_input=self.get_I_input(input_spike_trains,self.time)
        #self.U=self.update_U(torch.tensor(3))
        self.U = self.update_U(self.I_input)
        self.spike_train[self.time]=self.spike
        self.time+=1
        if self.time==self.max_spikes:
            self.adjust_synapse(input_spike_trains)

        return self.I_input, self.U

    ###########################################################################
    #   Our neuron is multi_input to one output , so we need to define the number
    #   of the input_synapse. What's more , according to the formula :
    #   I(t) = ∑i(wi*∑f(Isyn(t-tif)))
    #   Isyn(t) = q/tau_syn*exp(-t/tau_syn)
    ###########################################################################
    def get_I_input(self,spike_trains,time):
        ## We get the spike time in each synapse
        ## Also , the spike after our time will be reset to 0
        index=torch.arange(self.max_spikes).view([self.max_spikes,1])
        spike_trains=torch.where(spike_trains,torch.tensor(1),torch.tensor(0))
        spike_time=(index*spike_trains)
        spike_time[time:]=torch.tensor(0)
        spike_time=spike_time.float()
        ############################################################

        ## We should tranfer the spike trains to current
        I_syn=torch.exp((-(time-spike_time)/self.tau))*self.q/self.tau
        I_syn=torch.where(spike_time==0, torch.full_like(spike_time,0), I_syn)
        I_syn=torch.sum(I_syn,0)
        I=torch.sum(self.w*I_syn)
        ############################################################

        return I

    def update_U(self,I_input):
        ## During the intergrate : U'(t) = (R*I(t）-U(t)+Urest(t))/tau*dt + U(t)
        ## du = (R*I(t）-U(t)+Urest(t))/tau*dt
        du=(self.R*I_input-self.U+self.u_rest)/self.tau*self.dt
        self.U=du+self.U
        if self.U>=self.u_thresh:
            self.U=self.u_rest
            self.spike=True
        else:
            self.spike=False
        ############################################################
        return self.U
    def adjust_Intrinsic(self):
        if self.spike==True:
            drC=-42*10e-6*(1+1.2)*self.I_input
            drR=72*10e-6*(1+1.2)
            self.R= 1/(1/self.R+drR)
            self.C= 1/(1/self.C+drC)
        else:
            drC=(self.C+self.I_input*1.2)*10e-6*0.1
            drR=(1/self.R-1.2)*10e-6*0.1
            self.R = 1 / (1 / self.R + drR)
            self.C = 1 / (1 / self.C + drC)

    def adjust_synapse(self,input_spike_trains):
        output_spike_index = torch.nonzero(self.spike_train)
        input_spike_index0 = torch.nonzero(input_spike_trains.view([input_spike_trains.shape[1],input_spike_trains.shape[0]]))
        input_spike_index = input_spike_index0[:, 1]
        output_spike_index = output_spike_index.expand(output_spike_index.shape[0], input_spike_index.shape[0])
        input_spike_index1 = input_spike_index.expand(output_spike_index.shape[0], input_spike_index.shape[0])
        delta_t = (output_spike_index - input_spike_index1).float()
        delta_w0 = torch.where(delta_t >= 0, torch.exp(-delta_t), -torch.exp(delta_t))
        sun = input_spike_index0[:, 0].numpy().tolist()
        count = 0
        delta_w = []
        for t in range(self.n_syn):
            delta_w.append(torch.sum(delta_w0[:, count:count + sun.count(t)]))
            count += sun.count(t)
        self.w+=torch.tensor(delta_w)











'''

##  This is test code to check Neuron function above
##################################################################################################
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

##  We offer the spike trains as the poisson distribution
for step in range(steps):
    r = np.random.uniform(0, 1, size=(n_syn))
    syn_has_spiked[step, :] = r < f * dt * 1e-3

# We define the synaptic efficacy as a random vector
W = np.random.normal(1.0, 0.5, size=n_syn)
# Output variables
I = []
U = []
##################################################################################################
neuron=Neuron(max_spikes=T,w=W,n_syn=n_syn)
for time in range(T):
    syn_has_spiked=Variable(torch.tensor(syn_has_spiked))
    i,u = neuron.forward(syn_has_spiked,spike_time=time)
    I.append(i)
    U.append(u)
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

'''














class Neuron_input_I(object):
    def __init__(self, R=2, C=0.6, u_rest=-65, dt=1,
                 u_thresh=-50, T_rest=4.0,
                 n_syn=10, max_spikes=50, q=1.5, tau_syn=10.0
                 ):
        ###########################################################################
        #   During the intergrate : U'(t) = (R*I(t）-U(t)+Urest(t))/tau*dt + U(t)
        ###########################################################################
        # Membrane resistance in Ohm
        self.R = R
        self.C = C
        # Membrane resting potential in mV
        self.u_rest = u_rest
        # Membrane time constant in ms
        self.tau = self.R*self.C
        #Step in ms
        self.dt=dt

        ###########################################################################
        #   During the time at fire : We need to define a duration of resting time
        #   after refractory : T_rest .     Obviously , the threshold of U to fire
        #   is also needed .
        ###########################################################################
        # Membrane threshold potential in mV
        self.u_thresh = u_thresh
        # Duration of the resting period in ms
        self.T_rest = T_rest

        ###########################################################################
        #   Our neuron is multi_input to one output , so we need to define the number
        #   of the input_synapse. What's more , according to the formula :
        #   I(t) = ∑i(wi*∑f(Isyn(t-tif)))
        #   Isyn(t) = q/tau_syn*exp(-t/tau_syn
        ###########################################################################
        # Number of synapses
        self.n_syn = n_syn
        # Maximum number of spikes we remember
        self.max_spikes = max_spikes
        # The neuron synaptic 'charge'
        self.q = q
        # The synaptic time constant (ms)
        self.tau_syn = tau_syn
        # The synaptic efficacy
        self.w = torch.FloatTensor(np.random.normal(1.0, 0.5, size=self.n_syn))

        ###########################################################################
        #   Other variable that we need to define
        ###########################################################################
        self.U=self.u_rest
        self.spike_train=torch.tensor(np.full(self.max_spikes,False))
        self.spike=False
        self.time=0
        self.spike_count=0


    ###########################################################################
    #  Usually , we get the spike trains as the input . But our neuron is
    #  current_friendly . So we should transfer the spike trains
    #  to current(I) that we can deal with it by our neuron .
    ###########################################################################
    def forward(self,input_I):
        self.I_input=input_I
        #self.U=self.update_U(torch.tensor(3))
        self.U = self.update_U(self.I_input[self.time])
        self.spike_train[self.time]=self.spike
        self.time+=1

        #self.adjust_Intrinsic()
        return self.U


    def update_U(self,I_input):
        ## During the intergrate : U'(t) = (R*I(t）-U(t)+Urest(t))/tau*dt + U(t)
        ## du = (R*I(t）-U(t)+Urest(t))/tau*dt
        du=(self.R*I_input-self.U+self.u_rest)/self.tau*self.dt
        self.U=du+self.U
        #print(self.U)
        if self.U>=self.u_thresh:
            self.U=self.u_rest
            self.spike=True
            self.spike_count+=1
        else:
            self.spike=False
        ############################################################
        return self.U


    def adjust_Intrinsic(self):
        if self.spike==True:
            drC=-42*10e-6*(1+1.2)*self.I_input
            drR=72*10e-6*(1+1.2)
            self.R= 1/(1/self.R+drR)
            self.C= 1/(1/self.C+drC)
        else:
            drC=(self.C+self.I_input*1.2)*10e-6*0.1
            drR=(1/self.R-1.2)*10e-6*0.1
            self.R = 1 / (1 / self.R + drR)
            self.C = 1 / (1 / self.C + drC)


    def update_weight(self):




        return True













