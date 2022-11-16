# StyleGAN-pytorch-lightining

This repository contains an attempt to implement the StyleGAN V1 architecture using Pytorch and Pytorch Lightning. It have been proposed in the following paper:

> A Style-Based Generator Architecture for Generative Adversarial Networks \
Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA) \
http://stylegan.xyz/paper \
**Abstract**: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces

The official TensorFlow implementation can be found here: https://github.com/NVlabs/stylegan \
A new improved version called StyleGAN v2 was introduced in 2020:
> Training Generative Adversarial Networks with Limited Data \
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila

The official Tensorflow implementation can be found here https://github.com/NVlabs/stylegan2-ada/
The authors also released the Pytorch implementation: https://github.com/NVlabs/stylegan2-ada/

## How to use
---
To make things simple, I recommend using *Poetry* as the python package manager. It can be installed [here](https://python-poetry.org/docs/#installation).
After cloning the repo, you can install everything by doing:

> `poetry install`

For more details about the dependencies, please take a look at the [toml](pyproject.toml) file. 
Note that all the configs can be parametrized using the [config.yaml](configs/config.yaml) file. You can change path to your dataset as well as the hyperparameters of the model. 
To start the training:
> `python scripts/train.py fit --config configs/config.yaml`

## Content
---
This repo implements the following features:
- Implemented features related to the architecture
    -  Progressive Growing Training
    - Equalized Learning Rate
    - PixelNorm Layer
    - Exponential Moving Average
    - Minibatch Standard Deviation Layer
    - Style Mixing Regularization
    - Truncation Trick
    - Conditional GAN
- Implemented features related to the training process
    - Lightning CLI 
    - Gradient Clipping
    - Multi-GPU Training

## Acknowledgement
---
Most of the custom layers/ basic blocks were either inspired or copied (with variable renaming , clear documentation and more organization) from this [repo](https://github.com/lernapparat/lernapparat/tree/master/style_gan). I would like to thank the authors of this repo for helping me implementing the paper.

