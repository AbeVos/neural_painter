import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from math import log10
from statistics import mean
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor
from architectures.gan import Generator, Discriminator
from train_vae import save_sample_plot


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_samples(samples, path, color=None):
    if samples.shape[1] == 4:
        samples = samples[:, :3] * samples[:, -1].unsqueeze(1)

    save_image(samples, path, nrow=8)


def update_discriminator(images, actions, G, D, optim_D, real_label, fake_label,
                         criterion=None, train=True):
    criterion = criterion or nn.BCEWithLogitsLoss()

    D.zero_grad()
    G.zero_grad()

    G.eval()
    D.train()

    action_fake = torch.rand(actions.shape).to(device)
    fake = G(action_fake).detach()
    pred_fake = D(fake, action_fake)
    loss_D_fake = criterion(pred_fake, fake_label)
    # loss_D_fake.backward()

    pred_real = D(images, actions)

    loss_D_real = criterion(pred_real, real_label)

    loss_D = loss_D_fake + loss_D_real

    if train:
        loss_D.backward()

        optim_D.step()

    return loss_D.item()


def update_generator(images, actions, G, D, optim_G, real_label, criterion=None,
                     recon_criterion=None, recon_weight=1, train=True):
    criterion = criterion or nn.BCEWithLogitsLoss()
    recon_criterion = recon_criterion or nn.MSELoss()

    D.zero_grad()
    G.zero_grad()

    G.train()
    D.eval()

    fake = G(actions)
    pred_fake = D(fake, actions)

    loss_G = criterion(pred_fake, real_label) \
        + recon_weight * recon_criterion(fake, images)

    if train:
        loss_G.backward()
        optim_G.step()

    psnr = 10 * log10(1 / nn.functional.mse_loss(fake, images).item())

    return loss_G.item(), psnr


def iter_epoch(dataset, G, D, optim_G, optim_D, device, batch_size,
               recon_func=nn.MSELoss(), recon_weight=1, disc_interval=10,
               sample_interval=10, train=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                            drop_last=train)

    criterion = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Create fixed noise for sampling images.
    action_sample = torch.rand(64, dataset.action_dim).to(device)

    psnr_mean = []

    for idx, data in enumerate(dataloader):
        images, actions = data['image'], data['action']
        images = images[:, -1].unsqueeze(1)
        images = 2 * images - 1
        images = images.to(device).float()

        actions = actions.to(device).float()

        ones = torch.ones((len(actions), 1)).to(device)
        zeros = torch.zeros((len(actions), 1)).to(device)

        # Train generator.
        loss_G, psnr = update_generator(
            images, actions, G, D, optim_G, ones, criterion, recon_func,
            recon_weight, train=train)

        if idx % disc_interval == 0:
            # Train discriminator.
            loss_D = update_discriminator(
                images, actions, G, D, optim_D, ones, zeros, criterion,
                train=train)
        else:
            loss_D = 0

        psnr_mean.append(psnr)

        if train and idx % sample_interval == 0:
            print(f"{idx:03d} | D: {loss_D:.05f}, G: {loss_G:.05f} | PSNR: {psnr}")

            G.eval()

            sample = G(action_sample)
            sample = (sample + 1) / 2
            colors = torch.ones((len(sample), 3, 64, 64)).to(device) \
                    * action_sample[:, -3:][..., None, None]
            sample = torch.cat((colors, sample), dim=1)
            save_sample_plot(sample, "samples/samples_gan_painter.png")

    return mean(psnr_mean)



def train(params):
    print(params)
    batch_size = int(params['batch_size'])

    recon_crit = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss(),
    }
    recon_crit = recon_crit[params['recon_crit']]

    recon_weight = params['recon_weight']
    disc_interval = params['disc_interval']
    sample_interval = params['sample_interval']
    device = params['device']

    generator = Generator(train_set.action_dim).to(device)
    generator.apply(weights_init)
    discriminator = Discriminator(train_set.action_dim).to(device)
    discriminator.apply(weights_init)

    optim_G = optim.Adam(generator.parameters(), lr=params['lr_G'])
    optim_D = optim.Adam(discriminator.parameters(), lr=params['lr_D'])

    for epoch in range(params['epochs']):
        print(f"Epoch {epoch}")
        psnr_train = iter_epoch(
            train_set, generator, discriminator, optim_G, optim_D,
            device, batch_size, recon_crit, recon_weight,
            disc_interval, sample_interval)

        psnr_valid = iter_epoch(
            valid_set, generator, discriminator, None, None,
            device, batch_size, recon_crit, recon_weight,
            disc_interval, sample_interval, train=False)

        print(f"Validation PSNR: {psnr_valid}")

        # Assume training went awry if both training set and validation
        # set have negative PSNR.
        if psnr_train < 0 and psnr_valid < 0:
            break

    return {
        'loss': - psnr_valid,
        'status': STATUS_OK,

        'model': generator
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_root', dest='train_root', type=str, default='calligraphy_100k',
        help="")
    parser.add_argument(
        '--valid_root', dest='valid_root', type=str,
        default='calligraphy_1k',
        help="")
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help="Number of epochs to train for.")
    parser.add_argument(
        '--batch_size', dest='batch_size', type=int, default=256,
        help="Batch size for training.")
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    parser.add_argument(
        '--sample_interval', dest='sample_interval', type=int, default=20,
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    train_set = BrushStrokeDataset('labels.csv', args.train_root,
                                   transform=ToTensor())
    valid_set = BrushStrokeDataset('labels.csv', args.valid_root,
                                   transform=ToTensor())

    search_space = {
        'epochs': 1 + hp.randint('epochs', 10),
        'batch_size': hp.quniform('batch_size', 64, 256, 1),
        'lr_G': hp.uniform('lr_G', 1e-6, 1e-3),
        'lr_D': hp.uniform('lr_D', 1e-6, 1e-3),
        'recon_crit': hp.choice('recon_crit', ['MSE', 'L1']),
        'recon_weight': hp.uniform('recon_weight', 0, 2),
        'disc_interval': 1 + hp.randint('disc_interval', 20),
        'sample_interval': args.sample_interval,
        'device': device
    }

    '''
    trials = Trials()

    best = fmin(fn=train, space=search_space, algo=tpe.suggest, max_evals=10,
                trials=trials)

    print(best)
    print(trials.results)

    '''
    result = train({
        'epochs': 5,
        'batch_size': 256,
        'lr_G': 1e-4,
        'lr_D': 1e-5,
        'recon_crit': 'MSE',
        'recon_weight': 1,
        'disc_interval': 10,
        'sample_interval': args.sample_interval,
        'device': device
    })
