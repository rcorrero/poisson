---
title: Poisson | Development
created: '2020-06-26T17:02:47.180Z'
modified: '2020-07-05T18:36:29.931Z'
---

# Poisson | Development

At the highest level this project encompasses the design, development, and implementation of a satellite imagery processing pipeline. This pipeline takes images of near-shore open seas and extracts statistics relevant to the study of the behavior of small- to medium-sized vessels and other objects. The focus is on small vessels because they are unlikely to use active reporting systems such as [VMS](https://en.wikipedia.org/wiki/Vessel_monitoring_system) or [AIS](). Consequently much of the illegal, unreported, and unregulated ([IUU](https://en.wikipedia.org/wiki/Illegal,_unreported_and_unregulated_fishing)) fishing activity globally is done by [smaller vessels](http://biblioimarpe.imarpe.gob.pe/bitstream/123456789/2328/1/THESIS%20final%20-%20post%20defense.pdf#page=159
).

## Architecture
The first milestone is to develop the satellite image processing pipeline, _panoptis_. The inputs to panoptis are raw satellite images (geotiff files). Panoptis processes these images, identifies vessels on the water, and creates a dateset containing vessel locations, sizes (length, width, area, bounding boxes, etc.) and timestamps associated with the time at which the image was captured. Panoptis can be thought of as three separate components strung together sequentially:

1. Image preprocessor
2. Vessel detector (the "model")
3. Data postprocessor

One and three are scaffolding which supports the main development, the model. The model itself is by far the most computationally complex part of poisson because it must identify vessels from raw satellite imagery. 

To create and train a model with acceptable performance we need a training socket, called _hacedor_. This handles the I/O for the model, as well as hyperparameter selection, performance analysis, and report generation (describing the performance of the model). 
