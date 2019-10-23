import torch
import torch.nn.functional as fn
import numpy as np
import math
from torchvision import transforms
from torchvision import datasets
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader






class FilterKernel:
	def __init__(self, window_size):
		self.window_size = window_size

	def __call__(self):
		pass

class Kernel():
	def __init__(self,window_size,kernel):
		self.kernel=kernel
	def __call__(self):
		kernel = torch.from_numpy(self.kernel)
		return kernel.float()




class DoGKernel(FilterKernel):
	def __init__(self, window_size, sigma1, sigma2):
		super(DoGKernel, self).__init__(window_size)
		self.sigma1 = sigma1
		self.sigma2 = sigma2

	# returns a 2d tensor corresponding to the requested DoG filter
	def __call__(self):
		w = self.window_size//2
		x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
		a = 1.0 / (2 * math.pi)
		prod = x*x + y*y
		f1 = (1/(self.sigma1*self.sigma1)) * np.exp(-0.5 * (1/(self.sigma1*self.sigma1)) * (prod))
		f2 = (1/(self.sigma2*self.sigma2)) * np.exp(-0.5 * (1/(self.sigma2*self.sigma2)) * (prod))
		dog = a * (f1-f2)
		dog_mean = np.mean(dog)
		dog = dog - dog_mean
		dog_max = np.max(dog)
		dog = dog / dog_max
		dog_tensor = torch.from_numpy(dog)
		return dog_tensor.float()


class Filter:
	# filter_kernels must be a list of filter kernels
	# thresholds must be a list of thresholds for each kernel
	def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
		tensor_list = []
		self.max_window_size = 0
		for kernel in filter_kernels:
			if isinstance(kernel, torch.Tensor):
				tensor_list.append(kernel)
				self.max_window_size = max(self.max_window_size, kernel.size(-1))
			else:
				tensor_list.append(kernel().unsqueeze(0))
				self.max_window_size = max(self.max_window_size, kernel.window_size)
		for i in range(len(tensor_list)):
			p = (self.max_window_size - filter_kernels[i].window_size)//2
			tensor_list[i] = fn.pad(tensor_list[i], (p,p,p,p))

		self.kernels = torch.stack(tensor_list)
		self.number_of_kernels = len(filter_kernels)
		self.padding = padding
		if isinstance(thresholds, list):
			self.thresholds = thresholds.clone().detach()
			self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
		else:
			self.thresholds = thresholds
		self.use_abs = use_abs

	# returns a 4d tensor containing the flitered versions of the input image
	# input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
	def __call__(self, input):
		output = fn.conv2d(input, self.kernels, padding = self.padding).float()
		if not(self.thresholds is None):
			output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
		if self.use_abs:
			torch.abs_(output)
		return output




class S1C1Transform:
    def __init__(self, filter, timesteps=15):
        self.grayscale = transforms.Grayscale()
        self.size = transforms.Resize((16, 16))
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = Intensity2Latency(timesteps)
        self.cnt = 0

    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt += 1
        image = self.size(image)
        image = self.to_tensor(self.grayscale(image)) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()


class Intensity2Latency:
	def __init__(self, number_of_spike_bins, to_spike=False):
		self.time_steps = number_of_spike_bins
		self.to_spike = to_spike

	# intencities is a tensor of input intencities (1, input_channels, height, width)
	# returns a tensor of tensors containing spikes in each timestep (considers minibatch for timesteps)
	# spikes are accumulative, i.e. spikes in timestep i are also presented in i+1, i+2, ...
	def intensity_to_latency(self, intencities):
		# bins = []
		bins_intencities = []
		nonzero_cnt = torch.nonzero(intencities).size()[0]

		# check for empty bins
		bin_size = nonzero_cnt // self.time_steps

		# sort
		intencities_flattened = torch.reshape(intencities, (-1,))
		intencities_flattened_sorted = torch.sort(intencities_flattened, descending=True)

		# bin packing
		sorted_bins_value, sorted_bins_idx = torch.split(intencities_flattened_sorted[0], bin_size), torch.split(
			intencities_flattened_sorted[1], bin_size)

		# add to the list of timesteps
		spike_map = torch.zeros_like(intencities_flattened_sorted[0])

		for i in range(self.time_steps):
			spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
			spike_map_copy = spike_map.clone().detach()
			spike_map_copy = spike_map_copy.reshape(tuple(intencities.shape))
			bins_intencities.append(spike_map_copy.squeeze(0).float())
		# bins.append(spike_map_copy.sign().squeeze_(0).float())

		return torch.stack(bins_intencities)  # , torch.stack(bins)

	# return torch.stack(bins)

	def __call__(self, image):
		if self.to_spike:
			return self.intensity_to_latency(image).sign()
		return self.intensity_to_latency(image)



def local_normalization(input, normalization_radius, eps=1e-12):
	# computing local mean by 2d convolution
	kernel = torch.ones(1,1,normalization_radius*2+1,normalization_radius*2+1,device=input.device).float()/((normalization_radius*2+1)**2)
	# rearrange 4D tensor so input channels will be considered as minibatches
	y = input.squeeze(0) # removes minibatch dim which was 1
	y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
	means = fn.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
	y = y/means # normalization
	# swap minibatch with channels
	y.squeeze_(1)
	y.unsqueeze_(0)
	return y


class CacheDataset(torch.utils.data.Dataset):
	r"""A wrapper dataset to cache pre-processed data. It can cache data on RAM or a secondary memory.

	.. note::

		Since converting image into spike-wave can be time consuming, we recommend to wrap your dataset into a :attr:`CacheDataset`
		object.

	Args:
		dataset (torch.utils.data.Dataset): The reference dataset object.
		cache_address (str, optional): The location of cache in the secondary memory. Use :attr:`None` to cache on RAM. Default: None
	"""
	def __init__(self, dataset, cache_address=None):
		self.dataset = dataset
		self.cache_address = cache_address
		self.cache = [None] * len(self.dataset)

	def __getitem__(self, index):
		if self.cache[index] is None:
			#cache it
			sample, target = self.dataset[index]
			if self.cache_address is None:
				self.cache[index] = sample, target
			else:
				save_path = os.path.join(self.cache_address, str(index))
				torch.save(sample, save_path + ".cd")
				torch.save(target, save_path + ".cl")
				self.cache[index] = save_path
		else:
			if self.cache_address is None:
				sample, target = self.cache[index]
			else:
				sample = torch.load(self.cache[index] + ".cd")
				target = torch.load(self.cache[index] + ".cl")
		return sample, target

	def reset_cache(self):
		r"""Clears the cached data. It is useful when you want to change a pre-processing parameter during
		the training process.
		"""
		if self.cache_address is not None:
			for add in self.cache:
				os.remove(add + ".cd")
				os.remove(add + ".cl")
		self.cache = [None] * len(self)

	def __len__(self):
		return len(self.dataset)


def target_transform(target):
    return target // 1


def dataloader(path,s1c1,batch_size,shuttle=True):
	datafolder = CacheDataset(ImageFolder(path, s1c1, target_transform=target_transform))

	trainset = DataLoader(datafolder, batch_size=batch_size, shuffle=shuttle)

	return trainset