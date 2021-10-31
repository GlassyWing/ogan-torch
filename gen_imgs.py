import torch
import torchvision as tv
from torchvision.utils import make_grid, save_image

from ogan.model import OGAN
from ogan.utils import add_sn

if __name__ == '__main__':
    z_dim = 512
    img_size = 128
    num_layers = 4
    max_num_channels = img_size * 4

    device = "cpu" if torch.cuda.is_available() else 'cpu'

    rows = 6
    cols = 6

    num_imgs = rows * cols

    ogan = OGAN(z_dim=z_dim,
                img_size=img_size,
                num_layers=num_layers,
                max_num_channels=max_num_channels)
    ogan.apply(add_sn)

    pretrained_dict = torch.load("checkpoints/ogan_ckpt_7_-1.779468.pth", map_location=device)
    model_dict = ogan.state_dict()

    # Fiter out unneccessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    ogan.load_state_dict(model_dict)

    ogan.to(device)
    with torch.no_grad():
        z = torch.randn(num_imgs, z_dim).to(device)
        imgs = ogan.generator(z)
        imgs = (imgs + 1) / 2 * 255

        save_image(imgs, "output/demo.jpg", nrow=rows, normalize=True)
