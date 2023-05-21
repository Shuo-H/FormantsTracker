import torch
import torch.nn as nn
import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy as np

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



def get_smoothing_kernel(kernel_size,kernel_sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 1
    kernel = scipy.ndimage.gaussian_filter(kernel, sigma=kernel_sigma)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    return kernel


def plot_heatmap(data, title):

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the heatmap
    heatmap = ax.imshow(data, cmap='hot', aspect='auto', origin='lower')

    # Add colorbar
    cbar = plt.colorbar(heatmap)
    # Set the y-axis direction to be reversed
    plt.gca().invert_yaxis()
    
    # Add labels, title, and ticks
    ax.set_xlabel('number of frames')
    ax.set_ylabel('frequency bin')
    ax.set_title(title)

    plt.savefig(title)