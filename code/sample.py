import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from VP import VP
from scoreModel import ScoreNet

# Config
device='cpu'
n_epochs = 50
batch_size = 64
lr = 1e-4
num_steps=1000
checkpoint_dir = './checkpoints/'

def plot_images(images):
    sample_grid = make_grid(images, nrow=int(np.sqrt(images.shape[0])))
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.cpu().permute(1, 2, 0).squeeze())
    plt.show()

def setup_for_sampling(sde_params):
    beta_min, beta_max = sde_params
    params_str = '{}_{}'.format(beta_min, beta_max)
    checkpoint_path = checkpoint_dir+'ckpt_{}_{}epochs_{}.pth'.format("fashionmnist", n_epochs, params_str)

    sde = VP(beta_min, beta_max, num_steps)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_proba=sde.marginal_proba))
    score_model = score_model.to(device)
    score_model.eval()
    if os.path.exists(checkpoint_path): 
        score_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("model loaded from checkpoint!")
    else:
        print("missing checkpoint! first train a model with params: {}".format(params_str))

    return sde, score_model

if __name__ == "__main__":
    # load trained diffusion model
    sde_params = None
    sde, score_model = setup_for_sampling(sde_params)

    # sample
    sampler = None
    samples = sampler(score_model, sde, batch_size, num_steps=1000)
    plot_images(samples)
