import torch
from torchvision import transforms

import os
import cv2
import numpy as np
from PIL import Image

from .model import module
from annotator.hed import HEDdetector
from annotator.util import annotator_ckpts_path


remote_model_path = "https://github.com/aidreamwin/sketch_simplification_pytorch/releases/download/model/model_gan.pth"


class SketchDetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "model_gan.pth")
        self.immean, self.imstd = 0.9664114577640158, 0.0858381272736797
        self.model = module.Net()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(remote_model_path, model_dir=os.path.dirname(model_path), progress=True)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.hed_func = HEDdetector()

    def __call__(self, pre_img):
        img = 255 - self.hed_func(pre_img)
        assert img.ndim == 2
        img = Image.fromarray(img).convert('L')
        w, h = img.size[0], img.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        data = ((transforms.ToTensor()(img) - self.immean) / self.imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
        data = data.float().cuda()
        with torch.no_grad():
            pred = self.model.cuda().forward(data).float()[0][0]
            pred = pred.detach().cpu().numpy()
            pred = cv2.resize(pred, (w, h))*255
            pred = pred.astype(np.uint8)
        return pred
