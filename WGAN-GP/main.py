'''
Author: Samrito
Date: 2023-03-14 19:55:47
LastEditors: Samrito
LastEditTime: 2023-03-14 20:45:18
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad

import torch.optim as optim

from model.wgan import *

from tqdm import tqdm
from einops import rearrange
import imageio


def gradient_penalty(real_x, rand_y, D):
    epsilon = torch.from_numpy(np.random.uniform(
        0, 1, (real_x.shape[0], 1))).float().to(real_x.device)
    X_inter = epsilon * real_x + (1 - epsilon) * rand_y
    X_inter = Variable(X_inter, requires_grad=True).to(real_x.device)
    X_prob = D(X_inter)
    gradient = grad(outputs=X_prob,
                    inputs=X_inter,
                    grad_outputs=torch.ones_like(X_prob, device=X_prob.device),
                    create_graph=True,
                    retain_graph=True)[0]
    gradient = gradient.view(real_x.shape[0], -1)
    gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=-1))
    return 10 * (F.relu((gradient_norm - 1))).mean()  # 10 is gp-weight


def G_loss(real_x, G: Generator, D: Discriminator):
    size = (real_x.shape[0], real_x.shape[1], G.random_dim)
    device = real_x.device
    rand_x = G.sample_generator(*size).to(device)
    rand_y = G(rand_x)
    g_loss = D(rand_y).mean()
    return g_loss


def D_loss(real_x, G: Generator, D: Discriminator):
    size = (real_x.shape[0], real_x.shape[1], G.random_dim)
    device = real_x.device
    rand_x = G.sample_generator(*size).to(device)
    rand_y = G(rand_x)
    grad_pen = gradient_penalty(real_x, rand_y, D)
    d_loss = D(real_x).mean() - D(rand_y).mean() + grad_pen
    return d_loss


def train(dataset, num_epoch=1000000, save_img=True):
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=8)
    G = Generator(10, 28 * 28).to(device)
    D = Discriminator(28 * 28).to(device)
    G_optimizer = optim.Adam(params=G.parameters(), lr=1e-4)
    D_optimizer = optim.Adam(params=D.parameters(), lr=1e-4)

    def train_one_epoch(epoch_idx):
        D.train()
        G.train()
        databar = tqdm(dataloader)
        databar.desc = f"Epoch {epoch_idx}"
        databar.ncols = 100
        for i, data in enumerate(databar):
            feature, label = data
            feature = feature.to(device)
            label = label.to(device)
            feature = rearrange(feature, 'b c h w -> b c (h w)')
            # torch.autograd.set_detect_anomaly(True)
            d_loss = D_loss(feature, G, D)
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            if i % 5 == 0:
                g_loss = G_loss(feature, G, D)
                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()

            databar.set_postfix({
                'g loss': f'{g_loss:.4f}',
                'd loss': f'{d_loss:.4f}'
            })

    fixed_x = G.sample_generator(64, 1, 10).to(device)
    # training_progress_images = []
    try:
        for epoch in range(num_epoch):
            train_one_epoch(epoch)
            if save_img and epoch % 100 == 0:
                G.eval()
                generated_img = G(fixed_x).cpu()
                generated_img = rearrange(generated_img,
                                          'b c (h w) -> b c h w',
                                          h=28)
                save_image(generated_img,
                           f'output/training_{epoch}_epochs.png')
                # img_grid = make_grid(generated_img).numpy()
                # img_grid = rearrange(img_grid, 'c h w -> h w c')
                # img_grid = (255.0 * img_grid.squeeze()).astype(np.uint8)
                # training_progress_images.append(img_grid)
                # if epoch % 1000 == 0:
                #     imageio.mimsave(f'output/training_{epoch}_epochs.gif',
                #                     training_progress_images)
        torch.save(G, './checkpoint/Generator.pt')
        torch.save(D, './checkpoint/Discriminator.pt')
    except KeyboardInterrupt:
        torch.save(G, './checkpoint/Generator.pt')
        torch.save(D, './checkpoint/Discriminator.pt')


if __name__ == '__main__':
    data_train = datasets.MNIST('./datasets/',
                                transform=transforms.ToTensor(),
                                train=True,
                                download=True)
    train(data_train)