poisson &mdash; Richard Correro
==============================

poisson is a python module for small vessel detection in optical satellite imagery. This module provides a framework for training ship detection models and using them to identify vessels in satellite imagery from [planet](https://www.planet.com/). 

This repository is a work-in-progress. It will eventually contain the module itself, a trained model and its associated files, working notes from the module's creation, and several papers relevant to vessel detection methods.

Repository Structure
------------
```
.
├── LICENSE
├── README.md
├── notebooks
│   ├── coco_eval.py
│   ├── coco_utils.py
│   ├── engine.py
│   ├── hacedor-data-aug-01.ipynb
│   ├── hacedor-io-01.ipynb
│   ├── hacedor-io-02.ipynb
│   ├── maskrcnn_resnet50_fpn_01.ipynb
│   ├── transforms.py
│   └── utils.py
├── notes
│   ├── Pipeline\ ?\200\ Imagery\ Processing\ Pipeline.md
│   ├── Poisson\ ?\200\ Development.md
│   └── Trainer\ ?\200\ Training\ Socket\ for\ Vessel\ Detection\ Models.md
├── papers
│   ├── remotesensing-10-00511.pdf
│   └── vessel_detect_survey.pdf
├── poisson
│   ├── pipeline
│   │   ├── clip_imgs.py
│   │   ├── download_items.py
│   │   ├── make_cred_str.sh
│   │   └── make_itemlist.py
│   └── trainer
│       ├── coco_eval.py
│       ├── coco_utils.py
│       ├── engine.py
│       ├── transforms.py
│       ├── utils.py
│       └── vessel-mask-01.py
└── setup.py

```    

Support
-----------
Poisson was developed with the support of a research grant from the Stanford University Department of Statistics.

------------
Created by Richard Correro in 2020. Contact me at rcorrero at stanford dot edu