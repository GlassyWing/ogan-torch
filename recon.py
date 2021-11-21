import torch
from torchvision import transforms

from ogan.dataset import ImageFolderDataset
from ogan.model import OGAN
from ogan.utils import add_sn
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    z_dim = 512
    img_size = 128
    num_layers = 4
    max_num_channels = img_size * 4

    device = "cpu" if torch.cuda.is_available() else 'cpu'


    ogan = OGAN(z_dim=z_dim,
                img_size=img_size,
                num_layers=num_layers,
                max_num_channels=max_num_channels)
    ogan.apply(add_sn)

    pretrained_dict = torch.load("checkpoints/ogan_ckpt_4_ema_-0.039934.pth", map_location=device)
    model_dict = ogan.state_dict()

    # Fiter out unneccessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    ogan.load_state_dict(model_dict)
    ogan.to(device)

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    ds = ImageFolderDataset("G:/data/GAN/anime_face", img_size, transform=tfms)
    img = ds[385]

    plt.imshow(((img + 1) / 2).permute(1, 2, 0))
    plt.show()
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        z, _ = ogan.encoder(img)
        # z = (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)

        recon = ogan.generator(z)
        recon = (recon + 1) / 2 * 255
        recon = recon[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.imshow(recon)
    plt.show()
