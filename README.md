## Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

### [Project Page](https://shihaozhaozsh.github.io/unicontrolnet/) | [Paper (ArXiv)](https://arxiv.org/abs/2305.16322) 
<img width="800" alt="image" src="./figs/results.png">


## ⏳ : To Do
- [ ] Huggingface demo
- [ ] Release training code
- [x] Release test code
- [x] Release pre-trained models

## 💡 : Method
<div align="center">
<img width="800" alt="image" src="./figs/pipeline.png">
</div>

Uni-ControlNet is a novel controllable diffusion model that allows for the simultaneous utilization of different local controls and global controls in a flexible and composable manner within one model. This is achieved through the incorporation of two adapters - local control adapter and global control adapter, regardless of the number of local or global controls used. These two adapters can be trained separately without the need for joint training, while still supporting the composition of multiple control signals. 

<div align="center">
<img width="600" alt="image" src="./figs/comparison.png">
</div>

Here are the comparisons of different controllable diffusion models. N is the number of conditions. Uni-ControlNet not only reduces the fine-tuning costs and model size as the number of the control conditions grows, but also facilitates composability of different conditions.

## ⚙ : Setup
First create a new conda environment

    conda env create -f environment.yaml
    conda activate unicontrol

Then download the [pretrained model](https://drive.google.com/file/d/1lagkiWUYFYbgeMTuJLxutpTW0HFuBchd/view?usp=sharing) and put it to `./ckpt/` folder. The model is built upon Stable Diffusion v1.5.

## 💻 : Test
You can launch the gradio demo by:

    python src/test/test.py
    
This command will load the downloaded pretrained weights and start the App. We include seven example local conditions: Canny edge, MLSD edge, HED boundary, sketch, Openpose, Midas depth, segmentation mask, and one example global condition: content. 

<div align="center">
<img width="800" alt="image" src="./figs/demo_conditions.png">
</div>

You can first upload a source image and our code automatically detects its sketch. Then Uni-ControlNet generates samples following the sketch and the text prompt which in this example is "Robot spider, mars". The results are shown at the bottom of the demo page, with generated images in the upper part and detected conditions in the lower part:

<div align="center">
<img width="800" alt="image" src="./figs/demo_results.png">
</div>

You can further detail your configuration in the panel：

<div align="center">
<img width="800" alt="image" src="./figs/demo_panel.png">
</div>

Uni-ControlNet also handles multi-conditions setting well. Here is an example of the combination of two local conditions: Canny edge of the Stormtrooper and the depth map of a forest. The prompt is set to "Stormtrooper's lecture in the forest" and here are the results:

<div align="center">
<img width="800" alt="image" src="./figs/demo_results2.png">
</div>

With Uni-ControlNet, you can go even further and incorporate more conditions. For instance, you can provide the local conditions of a deer, a sofa, a forest, and the global condition of snow to create a scene that is unlikely to occur naturally. The prompt is set to "A sofa and a deer in the forest" and here are the results.

<div align="center">
<img width="800" alt="image" src="./figs/demo_results3.png">
</div>

## ☕️ : Training

Coming soon!

## 🎉 : Acknowledgments:

This repo is built upon [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main) and really thank to their great work!

## 📖 : Citation

```bibtex
@article{zhao2023unicontrolnet,
  title={Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models},
  author={Zhao, Shihao and Chen, Dongdong and Chen, Yen-Chun and Bao, Jianmin and Hao, Shaozhe and Yuan, Lu and K. Wong, Kwan-Yee},
  journal={arXiv preprint arXiv:2305.16322},
  year={2023}
}
```
