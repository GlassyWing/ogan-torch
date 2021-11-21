import argparse
import logging
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ogan.dataset import CifarDataset
from ogan.model import OGAN
from ogan.utils import correlation, weights_init, add_sn, update_average, toggle_grad
from ogan.augments import DiffAugment
from functools import partial

if __name__ == '__main__':

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for OGAN model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each sample batch")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size
    device = "cuda:0"

    lr = 2e-4
    z_dim = 256
    img_size = 32
    beta = 0.999
    num_layers = int(np.log2(img_size)) - 3
    max_num_channels = img_size * 8

    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    diff_aug = partial(DiffAugment, policy="color,translation,cutout")

    dataset = CifarDataset(opt.dataset_path, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    ogan = OGAN(z_dim, img_size, num_layers, max_num_channels, num_layers).to(device)
    ogan.apply(weights_init)
    ogan.apply(add_sn)

    ogan_shadow = deepcopy(ogan)
    ema_updater = update_average
    update_average(ogan_shadow, ogan, beta=0.)

    if opt.pretrained_weights is not None:
        pretrained_dict = torch.load(opt.pretrained_weights, map_location=device)
        model_dict = ogan.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        ogan.load_state_dict(model_dict)
        print("load pretrained weights!")

    encoder = ogan.encoder
    generator = ogan.generator

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.99))
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.99))

    step = 0
    x_fake_old = None
    for epoch in range(epochs):

        total_loss = 0
        total_size = 0
        train_bar = tqdm(dataloader)
        for i, x_real in enumerate(train_bar):
            x_real = x_real.to(device)
            z_in = torch.randn(x_real.shape[0], z_dim, device=device)


            """
            Train Encoder
            """

            toggle_grad(encoder, True)
            optimizer_e.zero_grad()
            x_fake = generator(z_in)
            x_fake_detach = x_fake.detach()
            z_fake, fake_inners = encoder(diff_aug(x_fake_detach))
            z_real, real_inners = encoder(diff_aug(x_real))
            z_fake_mean = torch.mean(z_fake, dim=1, keepdim=True)
            z_real_mean = torch.mean(z_real, dim=1, keepdim=True)

            inner_loss = 0.
            for fake_inner, real_inner in zip(fake_inners, real_inners):
                inner_loss = inner_loss + 0.5 * (real_inner.mean() - fake_inner.mean())

            z_corr = correlation(z_in, z_fake)
            qp_loss = 0.25 * (z_real_mean - z_fake_mean)[:, 0] ** 2 / torch.mean((x_real - x_fake_detach) ** 2,
                                                                                 dim=[1, 2, 3])
            e_loss = (torch.mean(0.5 * (z_real_mean - z_fake_mean) - z_corr)
                      + torch.mean(qp_loss)
                      + inner_loss
                      )

            e_loss.backward()
            optimizer_e.step()

            """
            Train Generator
            """

            toggle_grad(encoder, False)
            optimizer_g.zero_grad()
            z_fake, fake_inners = encoder(diff_aug(x_fake))
            z_fake_mean_ng = z_fake_mean.detach()
            z_fake_mean = torch.mean(z_fake, dim=1, keepdim=True)

            z_corr = correlation(z_in, z_fake)

            inner_loss = 0.
            for fake_inner in fake_inners:
                inner_loss = inner_loss + fake_inner.mean()

            # qp_loss = 0.25 * (z_fake_mean - z_real_mean.detach())[:, 0] ** 2 / torch.mean((x_real - x_fake) ** 2,
            #                                                                               dim=[1, 2, 3])

            g_loss = torch.mean(z_fake_mean - z_corr) + inner_loss
            g_loss.backward()
            optimizer_g.step()

            ema_updater(ogan_shadow, ogan, beta=beta)

            train_bar.set_description(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), e_loss.item(), g_loss.item())
            )

            step += 1

            if step != 0 and step % 50 == 0:
                with torch.no_grad():
                    z = torch.randn((64, z_dim)).to(device)
                    imgs = ogan.generator(z)
                    imgs = (imgs + 1) / 2 * 255

                    save_image(imgs, f"output/ae_ckpt_%d_%d.png" % (epoch, step,), nrow=8, normalize=True)

                    imgs = ogan_shadow.generator(z)
                    imgs = (imgs + 1) / 2 * 255
                    save_image(imgs, f"output/ae_ckpt_%d_%d_ema.png" % (epoch, step,), nrow=8, normalize=True)

            total_loss += (e_loss.item() + g_loss.item())
            total_size += x_real.shape[0]

        torch.save(ogan.state_dict(), f"checkpoints/ogan_ckpt_%d_%.6f.pth" % (epoch, total_loss / total_size))
        torch.save(ogan_shadow.state_dict(),
                   f"checkpoints/ogan_ckpt_%d_ema_%.6f.pth" % (epoch, total_loss / total_size))
