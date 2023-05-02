# AlexNet

> Keywords: ImageNet, DeepLearning, Convolution

It's introduction is more like a technical report instead of an introduction.

AlexNet does not have a conclusion section but a discussion section which is rare.

[video](https://www.youtube.com/watch?v=wYmlILPsLlY&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I&index=2) by Li Mu.

## Introduction

### Discussion Section

Unsupervised pre-training: use some un-labeled image data to set the weights properly at first then continue the supervised training (warming up). Because training a net back in that time was not easy.
AlexNet influenced the whole deep learning mindset, because at that time people thought the deep learning network can learn the pattern autonomously from a unlabeled dataset. At that time people were pursuing unsupervised learning since it's closer to human's perception.
Until recent years, BERT brought people's interests back to unsupervised learning.

### Feature Vector Extraction

The output vector from the network has good quality in the semantic space, because it extracts important features such that simple classifiers can put similar images together. This is by far the most important results from the paper.

### Datasets

ILSVRC-2010, ImageNet (8.9m images, 10,184 categories, 2009)

## Details

Major impacts:

1. e2e, no need to do feature extraction before hand
2. use RuLU (non-saturating nonlinearity), `f(x) = max(0, x)` instead of sigmoid, faster learning rate(not accurate anymore nowadays), simple enough
3. (not so important engineering details): cutting the network in two pieces to be put on two small GPUs.
4. not being recognized as important by the author, but the end 4096-vector can capture the semantic information in the image very well
   1. DL can be thought as an information compression process, it transfers the images, texts, etc to a vector which computer can understand

### Comments

Learning rate starts from 0, as epochs go on, the learning rate goes up and then goes down again. In the past, the learning rate is set manually during training, for instance, when operator observes that the training loss is not going down, we people manual tune it down.
