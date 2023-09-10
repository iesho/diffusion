import os
import torch
import torchvision.transforms as transforms
from tqdm import trange
from torchvision.datasets import FashionMNIST
from torch.optim import Adam
from torch.utils.data import DataLoader

from VP import VP
from scoreModel import ScoreNet

def loss_fn(model, x, sde, eps=1e-5):
    """ Inputs:
          model: score model (i.e. diffusion model)
          x: batch of images
          sde: instance of VP class
          eps: parameter for numerical stability (1e-5 for learning, 1e-3 for sampling)
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_proba(x, random_t)
    perturbed_x = mean + z * std[:, None, None, None]

    # predict the score function for each perturbed x in the batch and its corresponding random t
    score = model(perturbed_x, random_t)
    
    # compute loss
    losses = score * std[:, None, None, None] + z
    loss = torch.mean(torch.sum(losses**2, dim=(1,2,3)))
    return loss

def train(sde_params):
    # VP sde
    beta_min, beta_max = sde_params
    sde = VP(beta_min, beta_max, num_steps)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_proba=sde.marginal_proba))
    score_model = score_model.to(device)
    optimizer = Adam(score_model.parameters(), lr=lr)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    params_str = '{}_{}'.format(beta_min, beta_max)
    checkpoint_path = checkpoint_dir+'ckpt_{}_{}epochs_{}.pth'.format("fashionmnist", n_epochs, params_str)

    # load checkpoint if existing
    if os.path.exists(checkpoint_path): 
        score_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("model loaded from checkpoint!")
    # otherwise train from scratch
    else:
        losses = []
        patience = 0
        tqdm_epoch = trange(n_epochs)
        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            for x, y in train_loader:
                x = x.to(device)    
                loss = loss_fn(score_model, x, sde)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0] 
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            losses.append(avg_loss/ num_items)

            # TODO: save model state dictionary at checkpoint_path
            pass
            # TODO: Experiment with early stopping and patience
            pass
            pass
        
        # TODO: Plot your losses (log-scale) over epochs. Title your plot.
        pass

if __name__ == "__main__":
    # Config
    device='cpu'
    n_epochs = 50
    batch_size = 64
    lr = 1e-4
    num_steps=1000
    checkpoint_dir = './checkpoints/'

    # Dataset
    train_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST('.', train=True, transform=train_transforms, download=True);
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    sde_params = None
    train(sde_params)
