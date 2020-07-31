import re
from pathlib import Path

import torch
from torchvision import transforms

from neural_style import utils
from neural_style.transformer_net import TransformerNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## put your original images in this directory
input_dir = Path('./inputs')

with torch.no_grad():
    style_model = TransformerNet()
    state_dict = torch.load('nausicaa3.model')
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(device)

    for imagefile in input_dir.glob('*.jpg'):

        content_image = utils.load_image(imagefile, scale=None)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        output = style_model(content_image).cpu()

        # styled images will be saved to this directory
        utils.save_image(f'./outputs/{imagefile.name}', output[0])
