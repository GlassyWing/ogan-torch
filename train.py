import argparse
import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ogan.dataset import ImageFolderDataset
from ogan.model import OGAN
from ogan.utils import correlation, add_sn, weights_init

if __name__ == '__main__':

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for OGAN model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each sample batch")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size
    device = "cuda:0"

    lr = 2e-4
    z_dim = 256
    img_size = 64
    num_layers = 4

    dataset = ImageFolderDataset(opt.dataset_path, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    ogan = OGAN(z_dim, img_size, num_layers).to(device)
    ogan.apply(weights_init)

    if opt.pretrained_weights is not None:
        ogan.load_state_dict(torch.load(opt.pretrained_weights, map_location=device), strict=False)

    encoder = ogan.encoder
    generator = ogan.generator

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=15, eta_min=5e-4)
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    e_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e, T_max=15, eta_min=5e-4)

    step = 0
    for epoch in range(epochs):

        total_loss = 0
        total_size = 0
        for i, x_real in enumerate(dataloader):
            x_real = x_real.to(device)
            z_in = torch.randn(x_real.shape[0], z_dim).to(device)

            """
            Train Encoder
            """
            optimizer_e.zero_grad()
            x_fake = generator(z_in).detach()
            z_fake = encoder(x_fake)
            z_real = encoder(x_real)
            z_fake_mean = torch.mean(z_fake, dim=1, keepdim=True)
            z_real_mean = torch.mean(z_real, dim=1, keepdim=True)

            z_corr = correlation(z_in, z_fake)
            qp_loss = 0.25 * (z_fake_mean - z_real_mean)[:, 0] ** 2 / torch.mean((x_real - x_fake) ** 2, dim=[1, 2, 3])
            e_loss = torch.mean(z_fake_mean - z_real_mean - 0.5 * z_corr) + torch.mean(qp_loss)

            e_loss.backward()
            optimizer_e.step()

            """
            Train Generator
            """
            optimizer_g.zero_grad()
            x_fake = generator(z_in)
            z_fake = encoder(x_fake)
            z_fake_mean = torch.mean(z_fake, dim=1, keepdim=True)

            z_corr = correlation(z_in, z_fake)
            g_loss = - torch.mean(z_fake_mean + 0.5 * z_corr)
            g_loss.backward()
            optimizer_g.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), e_loss.item(), g_loss.item())
            )

            step += 1

            if step != 0 and step % 100 == 0:
                with torch.no_grad():
                    z = torch.randn((1, z_dim)).to(device)
                    gen_img = generator(z)
                    gen_img = gen_img.permute(0, 2, 3, 1)
                    gen_img = (gen_img[0].cpu().numpy() + 1) / 2  * 255
                    gen_img = np.round(gen_img, 0).astype(np.uint8)

                    plt.imshow(gen_img)
                    # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
                    plt.show()

            total_loss = (e_loss.item() + g_loss.item()) * x_real.shape[0]
            total_size += x_real.shape[0]

        g_scheduler.step()
        e_scheduler.step()

        torch.save(ogan.state_dict(), f"checkpoints/ogan_ckpt_%d_%.6f.pth" % (epoch, total_loss / total_size))
