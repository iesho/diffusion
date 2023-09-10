import torch
from tqdm import tqdm
import numpy as np

def Euler_Maruyama_sampler(score_model,
                           sde,
                           batch_size, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):
    # TODO: compute std at t=1 
    pass
    pass

    std = np.sqrt(1 - np.exp(2*sde._c_t(1)))

    # TODO: sample a batch of x at t=1
    channels = 1 
    height = 28
    width = 28
    n_dist = torch.distributions.Normal(0, std)
    init_x = n_dist.sample((batch_size, channels, height, width))



    # TODO: create a sequence of time_steps from 1 to very smoll
    time_steps = torch.linspace(1, eps, num_steps).to(device)
    step_size = (time_steps[0] - time_steps[1]).to(device)

    # TODO: the magic! 
    x = init_x.to(device)
    #print('my x', x)
    with torch.no_grad():
        for time_step in tqdm(time_steps):  
            # TODO  
            z = torch.randn_like(x).to(device)
            if time_step < time_steps[-2]:
                f = -sde.drift(x, time_step.view(-1)) + (sde.diffusion(time_step)**2) * score_model(x, time_step.view(-1))
                x = x + f*step_size 
                #x = x - (sde.drift(x, time_step.view(-1)) - sde.diffusion(time_step)**2*score_model(x, time_step.view(-1)))*step_size
            else:

              f = -sde.drift(x, time_step.view(-1)) + (sde.diffusion(time_step)**2) * (score_model(x, time_step.view(-1)))
              g2s = (sde.diffusion(time_step)**2) * (score_model(x, time_step.view(-1)))
              gtz = sde.diffusion(time_step)*torch.sqrt(step_size)*z
              x = x + f*step_size + gtz
              # x = x - (sde.drift(x, time_step) - sde.diffusion(time_step)**2*score_model(x, time_step.view(-1)))*step_size
              # + sde.diffusion(time_step)*torch.sqrt(step_size)*z
            #x = x_prev
            x_ = x  
          
    # Do not include any noise in the last sampling step.
    return x_


def predictor_corrector_sampler(score_model, 
                                sde,
                                batch_size,
                                num_steps=1000,
                                device='cuda',
                                snr=0.16,
                                num_corrector_steps=1,
                                eps=1e-3):
    # TODO: compute std at t=1 
    pass
    pass
    std = np.sqrt(1 - np.exp(2*sde._c_t(1)))

    # TODO: sample a batch of x at t=1
    channels = 1 
    height = 28
    width = 28
    n_dist = torch.distributions.Normal(0, std)
    init_x = n_dist.sample((batch_size, channels, height, width))


    # TODO: create a sequence of time_steps from 1 to very smoll
    time_steps = torch.linspace(1, eps, num_steps).to(device)
    step_size = time_steps[0] - time_steps[1].to(device)

    # TODO: the magic!
    x = init_x.to(device)
    q = 0
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            # TODO: setup
            #alpha = torch.exp(sde._c_t(time_step))
            pass

            # Corrector step (Langevin MCMC - alorithm 5 in [Song21])

            for j in range(num_corrector_steps):
                z = (torch.randn_like(x)).to(device)
                g = (score_model(x, time_step.view(-1)))
                e = (2*sde.alphas[q]*(snr*torch.norm(z, p=2)/torch.norm(g, p=2))**2).to(device)
                
                x = x + e*g + torch.sqrt(2*e)*z
                


            # Predictor step (Euler-Maruyama)
            
            z = torch.randn_like(x).to(device)
            if time_step < time_steps[-2]:
                f = -sde.drift(x, time_step.view(-1)) + (sde.diffusion(time_step)**2) * score_model(x, time_step.view(-1))
                x = x + f*step_size 
                #x = x - (sde.drift(x, time_step.view(-1)) - sde.diffusion(time_step)**2*score_model(x, time_step.view(-1)))*step_size
            else:

              f = -sde.drift(x, time_step.view(-1)) + (sde.diffusion(time_step)**2) * (score_model(x, time_step.view(-1)))
              g2s = (sde.diffusion(time_step)**2) * (score_model(x, time_step.view(-1)))
              gtz = sde.diffusion(time_step)*torch.sqrt(step_size)*z
              x = x + f*step_size + gtz
              q += 1
              # x = x - (sde.drift(x, time_step) - sde.diffusion(time_step)**2*score_model(x, time_step.view(-1)))*step_size
              # + sde.diffusion(time_step)*torch.sqrt(step_size)*z
            #x = x_prev
            x_ = x    

    # Do not include any noise in the last sampling step.
    return x_