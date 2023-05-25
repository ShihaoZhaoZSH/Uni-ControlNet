## Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

### [Project Page](https://shihaozhaozsh.github.io/unicontrolnet/)
<img width="800" alt="image" src="./figs/results.png">

## ⏳ To Do
- [ ] Huggingface demo
- [ ] Release training code
- [x] Release testing code
- [x] Release pre-trained models

## 💡 Method
<img width="800" alt="image" src="./figs/pipeline.png">

Uni-ControlNet is a novel controllable diffusion model that allows for the simultaneous utilization of different local controls and global controls in a flexible and composable manner within one model. This is achieved through the incorporation of two adapters - local control adapter and global control adapter, regardless of the number of local or global controls used. These two adapters can be trained separately without the need for joint training, while still supporting the composition of multiple control signals. 

<img width="800" alt="image" src="./figs/comparison.png">

Here are the comparisons of different controllable diffusion models. N is the number of conditions. Uni-ControlNet not only reduces the fine-tuning costs and model size as the number of the control conditions grows, but also facilitates composability of different conditions.

## ⚙ Setup
First create a new conda environment

    conda env create -f environment.yaml
    conda activate unicontrol

Then download the [pretrained model](https://drive.google.com/file/d/1lagkiWUYFYbgeMTuJLxutpTW0HFuBchd/view?usp=sharing) and put it to `./ckpt/` folder. The model is built upon Stable Diffusion v1.5.

## 💻 Test
You can launch the gradio demo by:

    python src/test/test.py
    
This command will load the downloaded pretrained weights and start the App. We include seven local conditions: Canny edge, MLSD edge, HED boundary, sketch, Openpose, Midas depth, segmentation mask, and one global condition: content. 

<img width="800" alt="image" src="./figs/demo_conditions.png">

We first upload a source image and our code automatically detects its sketch. Then Uni-ControlNet generates samples following the sketch and the text prompt which in this example is "Robot spider, mars". The results are shown at the bottom of the demo page, with generated images in the upper part and detected conditions in the lower part:

<img width="800" alt="image" src="./figs/demo_results.png">

You can further detail your configuration in the panel：

<img width="800" alt="image" src="./figs/demo_panel.png">

Uni-ControlNet also handles multi-conditions setting well. Here is an example of the combination of two local conditions: Canny edge of the Stormtrooper and the depth map of a forest. We set the prompt to "Stormtrooper's lecture in the forest" and here are the results:

<img width="800" alt="image" src="./figs/demo_results2.png">

With Uni-ControlNet, you can go even further and incorporate more conditions. For instance, we provide the local conditions of a deer, a sofa, a forest, and the global condition of snow to generate a scene that is unlikely to occur naturally. We set the prompt to "A sofa and a deer in the forest" and here are the results.

<img width="800" alt="image" src="./figs/demo_results3.png">

## ☕️ Training

Coming soon!

## 🎉 Acknowledgments:

This repo is built upon [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main) and really thank to their great work!