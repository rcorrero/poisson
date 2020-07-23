---
title: Pipeline | Imagery Processing Pipeline
created: '2020-06-25T20:23:26.524Z'
modified: '2020-07-21T23:35:03.195Z'
---

# Pipeline | Imagery Processing Pipeline

## Architecture
The image processing pipeline itself is written entirely in python using several third-party packages. As of now my intention is to build panoptis for use on a single machine until satisfactory performance is attained, at which point I will refactor the code, containerize it, and run it at scale using a container orchestration framework. Satellite image processing is [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) in that the task may be separated by area of interest (AOI), image type, band, etc. When searching for objects on near-shore open ocean large areas need to be analyzed, and this means that any interesting applications require large-scale image processing. To run the code on several replicates the code may be structured such that each instance processes images linked to from a database containing the list of images to be processed. Once the statistics of interest are obtained from the imagery, these results may be written to another database containing the results from all replicates. __Update:__ Initial calculations suggest that a trained model running on a single machine should be able to handle the largest AOIs we will need to label. Containerization is likely unecessary, but this is not certain.

## `image_splitter`
- Splits the input imagery into sections of the correct size for labeling.
- The size must be a parameter which the user may provide.


## `make_dataframe`
- Converts the outputs of the model into rows in a Pandas `Dataframe`. 


The pipeline consists of the following operations in order:
1. Accessing imagery from storage (Google Cloud Storage in my case)
2. Clipping the imagery to the AOI
3. Identifying land and other noise in the imagery (e.g. cloudcover)
4. Object detection using a set of techniques (so that objects of different sizes may be detected)
5. Object classification by size, shape, and other factors of interest
6. Writing detected object data to a database or dataframe 

The basic statistics we need for each detected object are its
- Location (Lat/Long or other coordinate system)
- Time (timestamp of image in which the object is detected)
- Size 
- Other classification data based on the above


__Update 2020/7/14:__ I've spent the last two weeks working on the vessel detection and segmentation algorithm in pytorch. My focus has been developing _hacedor_ and the model itself. Now that the training socket is functional (or at least _minimally-viable_), I'm returning to _panoptis_. The main develop job is to handle image I/O from GCS, passing images to the trained model, processing the results, and storing them in a dataframe/datatable.

### Sample Labeling 
The target samples are  obtained from [Planet](), each with four channels and of a fixed size. The AOI, however, may only inlcude part of the image, and as such the region actually provided to the model may be of arbitrary size. [The model architecture]() can handle images of different sizes. I'm unsure whether different numbers of channels are also acceptable.
 

## The Learning Problem
Achieving acceptable performance and generalizability requires framing vessel detection as a machine learning problem. 

As described by [this survey](https://doi.org/10.1016/j.rse.2017.12.033) the learning workflow is
1. Mask land using a coastline shapefile
2. Correct environmental distortions in the images and mask any areas covered by thick clouds
3. Using image processing techniques identify potential vessels
4. Using a trained discriminator label candidate vessels as `vessel` or `not vessel`

Steps three and four may be combined into a single discriminator which takes corrected images as input. The decision of whether to combine these steps will be made based on the structure of previous vessel detection algorithms. If it seems possible to train discriminators with acceptable levels of performance which do not require candidate detection then I'll do that.

The logical avenue for development is toward deeper models, but there is likely wisdom in begining with a shallow model and developing the infrastructure necessary to train it – the training socket. Once this is built, model refinement, and importantly, the development of deeper models may proceed easily and at the same level of abstraction: model design and implementation. By abstracting away the finicky details of image preprocessing, I/O, etc., you can focus on designing models which yield better performance.

### Signal Source

[This dataset](https://www.iuii.ua.es/datasets/masati/) contains images of land and sea with seaborne vessels labeled with bounding boxes. [This dataset](https://www.kaggle.com/c/airbus-ship-detection/overview) contains roughly a quarter-million land and sea images with similar labels, of various resolutions and clearly gathered from several different imaging platforms. The latter lacks the clear catagorization which the former sports, but its being much larger makes it more attractive as a first training set. 



