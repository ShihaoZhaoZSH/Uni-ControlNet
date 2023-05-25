## Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

### [Project Page](https://shihaozhaozsh.github.io/unicontrolnet/)
<img width="800" alt="image" src="./figs/results.png">

## To Do
- [ ] Huggingface demo
- [ ] Release training code
- [x] Release testing code
- [x] Release pre-trained models

## Method
<img width="800" alt="image" src="./figs/pipeline.png">

Uni-ControlNet is a novel controllable diffusion model that allows for the simultaneous utilization of different local controls and global controls in a flexible and composable manner within one model. This is achieved through the incorporation of two adapters - local control adapter and global control adapter, regardless of the number of local or global controls used. These two adapters can be trained separately without the need for joint training, while still supporting the composition of multiple control signals.

## Setup
First create a new conda environment

    conda env create -f environment.yaml
    conda activate unicontrol

Then download the [pretrained model](https://drive.google.com/file/d/1lagkiWUYFYbgeMTuJLxutpTW0HFuBchd/view?usp=sharing) and put it to `./ckpt/` folder. The model is built upon Stable Diffusion v1.5.

## Test
You can launch the gradio demo by:

    python src/test/test.py
    
This command will load the downloaded pretrained weights and start the App. You can load source images for each conditions:

<img width="800" alt="image" src="./figs/demo_conditions.png">

The results are shown at the bottom of the demo page, with generated images in the upper part and detected conditions in the lower part.

<img width="800" alt="image" src="./figs/demo_results.png">

You can further detail your configuration in the panel：

<img width="800" alt="image" src="./figs/demo_panel.png">

## Training

Coming soon!

## Acknowledgments:

This repo is built upon [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main) and really thank to their great work!
