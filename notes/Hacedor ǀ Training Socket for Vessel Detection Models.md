---
title: Hacedor | Training Socket for Vessel Detection Models
created: '2020-07-05T18:13:08.397Z'
modified: '2020-07-07T17:17:53.948Z'
---

# Hacedor | Training Socket for Vessel Detection Models

Hacedor contains all of the code necessary to train vessel detection models using labeled datasets. [This dataset](https://www.iuii.ua.es/datasets/masati/) contains images of land and sea with seaborne vessels labeled with bounding boxes. [This dataset](https://www.kaggle.com/c/airbus-ship-detection/overview) contains roughly a quarter-million land and sea images with similar labels, of various resolutions and clearly gathered from several different imaging platforms. The latter lacks the clear catagorization which the former sports, but its being much larger makes it more attractive as a first training set.

As a working document, this records the hacedor development process. This can be viewed roughly as the creation of two objects, each inseparable from the other: a trained vessel detection model, and the structural code necessary to train it. 

## Theory

Previous attempts to design hand-engineered models for vessel detection have yielded mixed results. Even though vessel detection is relatively simple, the representations necessary to accurately identify vessels in diverse imagery clearly warrant complex models. After trying to design unsupervised vessel detection methods I realized that the amount of hand-engineering necessary to make something functional without more sophisticated algorithms was prohibitive. I found [these](https://www.sciencedirect.com/science/article/pii/S0034425717306193?via%3Dihub#f0005) [papers](https://www.mdpi.com/2072-4292/10/4/511/pdf) which led me to the labeled datasets. The first paper is a survey of ship detection methods. It explains precisely why models with expert-crafted features are difficult and the advantages of more statistically-sophisticated methods. The only reason to prefer more basic image-processing methods would be because there is no coverage of the objects we wish to label in available supervised datasets. I originally believed this to be the case, but based on what I've read in the last week, these two datasets provide acceptable coverage for small vessels, and other researchers have had success with learning tasks similar to mine using them. 

Therefore I begin with a survey of the theoretical foundations of neural networks and in particular convolutional neural networks (CNNs).

### Neural Networks – Foundations
I start with a review of [these deep learning notes](http://cs229.stanford.edu/notes2019fall/cs229-notes-deep_learning.pdf) from the class CS 229 at Stanford. The first quote worth presenting: "Part of the magic of a neural network is that all you need are the input features $x$ and the output $y$ while the neural network will figure out everything in the middle by itself. The process of a neural network learning the intermediate features is called _end-to-end learning_" [emphasis mine]. 

The rest of the notes discuss basic model conepts related to parameter initialization, optimization through backpropagation, etc. 

[These](https://cs231n.github.io/convolutional-networks/#architectures) notes from CS 231N more thoroughly cover CNNs and associated concepts. Discussing convolutional layers, these notes say, "First, the depth of the output volume is a hyperparameter: it corresponds to the number of filters we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a depth column (some people also prefer the term fibre)." Thus the third dimension in a convolutional layer gives the number of filters associated with that layer. The size of the first two dimensions (i.e. width and height) of a convolutional layer are determined by the stride length associated with the layer and the padding applied to the input.

Of pooling layers, these notes suggest that all-convolutional architectures may be preferable and that pooling is possibly out of favor among engineers: "Many people dislike the pooling operation and think that we can get away without it. For example, Striving for Simplicity: The All Convolutional Net proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers."

An important take-away from these notes comes near the end: "In practice: use whatever works best on ImageNet. If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch."

### The Task
Vessel detection requires that we may not only identify the presence/absence of vessels, but also the location of the vessels in the images. In this sense we require a model which may perform object segmentation (sometimes known as semantic segmentation). 

## Implementation

### Process
Heeding the advice in the CS 231N notes, I intend to use a pretrained CNN as the foundation on which to develop our task-specific model. This is [standard practice now](https://cs231n.github.io/transfer-learning/). 

The problem of different image sizes is apparently easily-solvable, according to these notes: "Due to parameter sharing, you can easily run a pretrained network on images of different spatial size. This is clearly evident in the case of Conv/Pool layers because their forward function is independent of the input volume spatial size (as long as the strides “fit”). In case of FC layers, this still holds true because FC layers can be converted to a Convolutional Layer: For example, in an AlexNet, the final pooling volume before the first FC layer is of size [6x6x512]. Therefore, the FC layer looking at this volume is equivalent to having a Convolutional Layer that has receptive field size 6x6, and is applied with padding of 0."

### Frameworks
#### `torch`
[This](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) page walks through the process of using a pre-trained classifier as the foundation for our specific object detection task in `torch`. [This](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/train.py) file contains a good reference for transfer learning approaches with `torch`. We can use a pre-trained [`torch`](https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn) [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) model. This model has a model has a ResNet-50-FPN backbone. Using [this](https://github.com/pytorch/vision/blob/v0.3.0/references/detection/train.py) file's code as a foundation we can train a model using the pretrained Mask R-CNN by simply removing the pretrained model's head (final layer), replacing it with our own (with `num_classes=2`) and then training it on the [Airbus dataset](https://www.kaggle.com/c/airbus-ship-detection/overview). Alternatively, we can use the [MASATI dataset](https://www.iuii.ua.es/datasets/masati/). Although the MASATI dataset is much too small to train a model from scratch, adjusting the weights of a pre-trained model using its ~7,000 segmented images is certainly possible, especially with the help of [`torchvision.transforms`](https://pytorch.org/docs/stable/torchvision/transforms.html).

#### `skorch`
Although I haven't utilized this module yet, I intend to design the structural code of hacedor as if the trained model conforms to the Sci-kit Learn API. This is for both my own benefit, as I'm intimately familiar with `sklearn`, but also because I believe doing so will abstract away much of the `torch` boilerplate which is unecessary in this case.

### Data Preparation
I begin by downloading the Airbus dataset and extracting the files. There are roughly a quarter-million `.jpg` files and the dataset is roughly 29 GB so this takes a bit of time. Because we're interested in identifying vessels of smaller size, I then attempt to identify the smallest labeled vessels in the dataset. The labels (or masks) are contained within a `.csv` file listing image ids with [run-length encodings](https://www.kaggle.com/inversion/run-length-decoding-quick-start) of the vessels in the image. Since each encoding explicitly lists the number of pixels contained within the bounding rectangle we can quantify the size distribution by pixels. I do exactly this, and then I plot the destribution. 

### Data Augmentation
It's clear that the coverage for the small end in this dataset is lacking. Therefore we need to augment the small end in some way. My first thought is to downsample the imagery. These downsampled samples can be used in conjunction with original samples. When downsampling, however, we must also adjust the labeled masks to match the smaller images.
