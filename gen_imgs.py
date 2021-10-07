import torch
import torchvision as tv
from torchvision.utils import make_grid, save_image

from ogan.model import OGAN

if __name__ == '__main__':
    z_dim = 128
    img_size = 128
    num_layers = 4
    max_num_channels = img_size * 8

    device = "cpu" if torch.cuda.is_available() else 'cpu'

    rows = 6
    cols = 6

    num_imgs = rows * cols

    ogan = OGAN(z_dim=z_dim,
                img_size=img_size,
                num_layers=num_layers,
                max_num_channels=max_num_channels)

    ogan.load_state_dict(torch.load("checkpoints/ogan_ckpt_4_0.096371.pth", map_location=device))

    ogan.eval()
    ogan.to(device)
    with torch.no_grad():
        z = torch.randn(num_imgs, z_dim).to(device)
        imgs = ogan.generator(z)
        imgs = (imgs + 1) / 2 * 255

        save_image(imgs, "output/demo.jpg", nrow=rows, normalize=True)
