import cv2
from PIL import Image

import torch
from transformers import AutoProcessor, CLIPModel

from annotator.util import annotator_ckpts_path


class ContentDetector:
    def __init__(self):

        model_name = "openai/clip-vit-large-patch14"

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=annotator_ckpts_path)

    def __call__(self, img):
        assert img.ndim == 3
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=img, return_tensors="pt").to('cuda')
            image_features = self.model.get_image_features(**inputs)
            image_feature = image_features[0].detach().cpu().numpy()
        return image_feature
