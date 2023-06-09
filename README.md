# FLAIR #1 : Semantic segmentation and domain adaptation challenge proposed by the French National Institute of Geographical and Forest Information (IGN).

Code : [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/master/LICENSE) &emsp; Dataset : [![license](https://img.shields.io/badge/License-IO%202.0-green.svg)](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)


Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. Get in touch at ai-challenge@ign.fr



![Alt bandeau FLAIR-IGN](images/flair_bandeau.jpg?raw=true)



<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">
  
<h2 style="margin-top:5px;">Links</h2>


- **Datapaper :** https://arxiv.org/pdf/2211.12979.pdf

- **Dataset links :** https://ignf.github.io/FLAIR/

- **Challenge page :**  https://codalab.lisn.upsaclay.fr/competitions/8769 [🛑 closed!]

</div>


## Context & Data

The FLAIR #1 dataset is sampled countrywide and is composed of over 20 billion annotated pixels, acquired over three years and different months (spatio-temporal domains). The dataset is available to download <a href="https://ignf.github.io/FLAIR/">here.</a> It consists of 512 x 512 patches with 13 (baselines) or 19 (full) semantic classes (see associated datapaper). Each patch has 5 channels (RVB-Infrared-Elevation). 

<br>

<figure>
  <img
  src="images/flair-1_spatiotemporal.png"
  alt="ortho image and train/test geographical repartition">
  <figcaption>ORTHO HR® aerial image cover of France (left), train and test spatial domains of the dataset (middle) and acquisition months defining temporal domains (right).</figcaption>
</figure>


<br>
<br>

<p align="center">
  <img width="70%" src="images/flair-1_patches.png">
  <br>
  <em>Example of input data (first three columns) and corresponding supervision masks (last column).</em>
</p>



<br>


## Baseline model 

A U-Net architecture with a pre-trained ResNet34 encoder from the pytorch segmentation models library is used for the baselines. The used architecture allows integration of patch-wise metadata information and employs commonly used image data augmentation techniques. It has about 24.4M parameters and it is implemented using the _segmentation-models-pytorch_ library. The results are evaluated with an Intersection Over Union (IoU) metric and a single mIoU is reported (see associated datapaper).

The _metadata_ strategy refers encoding metadata with a shallow MLP and concatenate this encoded information to the U-Net encoder output. The _augmentation_ strategy employs three typical geometrical augmentations (see associated datapaper).


Example of a semantic segmentation of an urban and coastal area in the D076 spatial
domain, obtained with the baseline trained model:


<p align="center">
  <img width="100%" src="images/flair-1_predicted.png">
  <br>
  <em>Example of a semantic segmentation result using the baseline model.</em>
</p>


<br>

## Usage 

The `flair-1-config.yml` file controls paths, hyperparameters and computing ressources. The file `requirement.txt` is listing used libraries for the baselines.

To launch a training/inference/metrics computation, you can either use : 

- ```
  main.py --config_file=flair-1-config.yml
  ```

-  use the `flair-1-baseline.ipynb` notebook guiding you through data visualization, training and testing steps.

A toy dataset (reduced size) is available to check that your installation and the information in the configuration file are correct.

<br>

## Leaderboard

| Model | mIoU 
------------ | ------------- 
| baseline U-Net (ResNet34) | 0.5443±0.0014
| baseline U-Net (ResNet34) + _metadata + augmentation_ | 0.5570±0.0027

If you want to submit a new entry, you can open a new issue.
<b> Results of the challenge will be reported soon! </b>

The baseline U-Net with ResNet34 backbone obtains the following confusion matrix: 


<p align="center">
  <img width="50%" src="images/flair-1_heatmap.png">
  <br>
  <em>Baseline confusion matrix of the test dataset normalized by rows.</em>
</p>


## Reference
Please include a citation to the following article if you use the FLAIR #1 dataset:

```
@article{garioud2022flair1,
  doi = {10.13140/RG.2.2.30183.73128/1},
  url = {https://arxiv.org/pdf/2211.12979.pdf},
  author = {Garioud, Anatol and Peillet, Stéphane and Bookjans, Eva and Giordano, Sébastien and Wattrelos, Boris},
  title = {FLAIR #1: semantic segmentation and domain adaptation dataset},
  publisher = {arXiv},
  year = {2022}
}
```

## Acknowledgment
This work was performed using HPC/AI resources from
GENCI-IDRIS (Grant 2022-A0131013803).

## Dataset license

The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration. 
If you are looking for an English version of this license, you can find it on the official GitHub page at the [official github page](https://github.com/etalab/licence-ouverte).

As stated by the license :

### Applicable legislation

This licence is governed by French law.

### Compatibility of this licence

This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).
