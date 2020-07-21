---
title: Trainer | Training Socket for Vessel Detection Models
created: '2020-07-05T18:13:08.397Z'
modified: '2020-07-21T17:33:40.519Z'
---

# Trainer | Training Socket for Vessel Detection Models

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

`torch` requires a `Dataset` class for model I/O. I'm currently working on implementing a class which inherits from `Dataset` to handle I/O, along with data augmentation methods. Since [the dataset](https://www.kaggle.com/c/airbus-ship-detection/overview) was the subject of a data science competition there are several openly-available python notebooks containing similar I/O classes and methods out there. [This particular notebook](https://www.kaggle.com/windsurfer/baseline-u-net-on-pytorch) contains much relevant code and my intention is to build on his code.

##### `train` Function
Let's run through `train` from the notebook linked to above. 

```
# From https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn
# main train routine
# Implementation from  https://github.com/ternaus/robot-surgery-segmentation
def train(lr, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=1, fold=1):
```

The parameters are `lr` or learning rate (a `float`), `model`, an `object` whose class inherits from `torch.nn.Module`, a loss `function` for image segmentation, data loaders which return (image, mask) samples, a validation method which calculates loss by foward propagation of samples through the model, and an optimizer (Adam in the referenced implementation). 

```
    optimizer = init_optimizer(lr)
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
       
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))


    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
```

    Now begins the actual training loop which always consists of two nested `for` loops: the first iterating over epochs, and the latter over samples in the training dataset. 

```
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  BATCH_SIZE)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
```

Here `inputs` is a set of images of size `batch_size` and `targets` is the corresponding set of masks. 

```
                inputs, targets = variable(inputs), variable(targets)
```

Here `variable` converts `inputs` into a list of `torch.autograd.Variable` objects. Because `model` does not explicitly define how to handle batches of training samples, it appears that `torch.nn.Module` defines such functionality.

```
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
```

In terms of I/O, the most important object is the `Dataset`. Once this is completed I can focus on data augmentation. 

#### `skorch`
Although I haven't utilized this module yet, I intend to design the structural code of hacedor as if the trained model conforms to the Sci-kit Learn API. This is for both my own benefit, as I'm intimately familiar with `sklearn`, but also because I believe doing so will abstract away much of the `torch` boilerplate which is unecessary in this case.

### Data Preparation
I begin by downloading the Airbus dataset and extracting the files. There are roughly a quarter-million `.jpg` files and the dataset is roughly 29 GB so this takes a bit of time. Because we're interested in identifying vessels of smaller size, I then attempt to identify the smallest labeled vessels in the dataset. The labels (or masks) are contained within a `.csv` file listing image ids with [run-length encodings](https://www.kaggle.com/inversion/run-length-decoding-quick-start) of the vessels in the image. Since each encoding explicitly lists the number of pixels contained within the bounding rectangle we can quantify the size distribution by pixels. I do exactly this, and then I plot the distribution. 

### Data Augmentation
It's clear that the coverage for the small end in this dataset is lacking. Therefore we need to augment the small end in some way. My first thought is to downsample the imagery. These downsampled samples can be used in conjunction with original samples. When downsampling, however, we must also adjust the labeled masks to match the smaller images.

Rather than downsampling images, saving them, and adding them to the data set (thus manually "augmenting" the data), I believe the best option is to use `torch.transforms` classes. Specifically, I can define a custom `Downsample` tranform which implements a `__call__` method which simply takes both the sample image and mask and downsamples them by the desired factor. It should do so only _some_ of the time, and therefore should use `random` to determine whether to resize a sample.

#### `Transform`s
`RandomDownsample` is a `torch` transform which takes a sample from the kaggle training dataset and with a user-provided probability, downsamples the image and mask in that sample. 

### Peculiarities of `maskrcnn_resnet50_fpn`
Because I intend to train the production vessel detection model from a pretrained model, I must use follow the protocol defined by `maskrcnn_resnet50_fpn` which says

 """

    Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format,  with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,  with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)

    Mask R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    
    """
