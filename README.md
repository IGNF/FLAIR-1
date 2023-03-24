# Semantic segmentation and domain adaptation challenge proposed by the French National Institute of Geographical and Forest Information (IGN).

Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset.

## Important links

**Data paper:** https://arxiv.org/pdf/2211.12979.pdf

**Challenge page :** https://codalab.lisn.upsaclay.fr/competitions/8769#learn_the_details [!CLOSED]

**Dataset link :** : https://ignf.github.io/FLAIR-Challenges/


![Alt bandeau FLAIR-IGN](images/visuel_FLAIR_bandeau.jpg?raw=true)

## Context

We present here a large dataset ( >20 billion pixels) of aerial imagery, topographic information and land cover (buildings, water, forest, agriculture...) annotations with the aim to further advance research on semantic segmentation , domain adaptation and transfer learning. Countrywide remote sensing aerial imagery is by necessity acquired at different times and dates and under different conditions. Likewise, at large scales, the characteristics of semantic classes can vary depending on location and become heterogenous. This opens up challenges for the spatial and temporal generalization of deep learning models!

<figure style="text-align:center">
  <img
  src="images/FR_ortho_and_dataset.png"
  alt="ortho image and train/test geographical repartition">
  <figcaption>ORTHO HR® aerial image cover of France (left) and train and test spatial domains of the dataset (right).</figcaption>
</figure>


## Data

The FLAIR-one dataset consists of 77,412 high resolution (0.2 m spatial resolution) patches with 13 semantic classes (19 original classes remapped to 13, see the associated paper in the starting kit for explanation). The dataset covers a total of approximatly 800 km², with patches that have been sampled accross the entire metropolitan French territory to be illustrating the different climate and landscapes (spatial domains). The aerial images included in the dataset were acquired during different months and years (temporal domains).

<figure style="text-align:center">
  <img
  src="images/patches.png"
  alt=" patches examples">
  <figcaption>Example of input data (first three columns) and corresponding supervision masks (last column).</figcaption>
</figure>

## Baseline and challenge leaderboard

A U-Net architecture with a pre-trained ResNet34 encoder from the pytorch segmentation models library is used for the baselines. The used architecture allows integration of patch-wise metadata information and employs commonly used image data augmentation techniques. It has about 24.4M parameters and it is implemented using the _segmentation-models-pytorch_ library. The results are evaluated with an Intersection Over Union (IoU) metric. More detailed results are presented in the technical description of the dataset.

### Baseline results 

| Model | mIoU 
------------ | ------------- 
| baseline | 0.5443±0.0014
| baseline + _bottom + augmentation_ | 0.5570±0.0027

The _bottom_ strategy refers to the adding a MLP encoded
metadata to the last layer of the architecture’s encoder. The
_augmentation_ strategy uses the three geometrical augmenta-
tions described in the data paper with a probability of 0.5.

Here is the confusion matrix obtained over the testing data.

<figure style="text-align:center">
  <img
  src="images/FLAIR-1_baseline_heatmap.png"
  alt="Confusion matrix">
  <figcaption>Baseline confusion matrix of the test dataset normalized by rows.</figcaption>
</figure>

And an example of a semantic
segmentation of an urban and coastal area in the D076 spatial
domain, obtained with the baseline trained model:

<figure style="text-align:center">
 <img
  src="images/image_pred_rvb.png"
  alt="Confusion matrix">
  <figcaption>Example of a semantic segmentation result using the baseline model</figcaption>
</figure>

### Challenge results

Here we will compile the results of the challenge.



# FLAIR-one baseline: starting-kit 

The starting-kit contains :

- toy_dataset_flair-one (folder): 
		sample of the full dataset with same structure and naming convention.

- metadata (folder): 
		contains a .json file with the metadata associated to provided patches.

-  py_module (folder): 
		contains .py files defining the modules used in the notebook.

- flair-one-baseline.ipynb (notebook): 
		a notebook allowing to launch the py_module, explore the data and train the baseline.


-------------------------------
### **How to use flair-one-baseline.ipynb** You can use this notebook in two different ways. Carefully read the following: 


### Option 1: locally

This option is more practical as the data is relatively volumineous. 
To do so, you just need to download the whole content of the starting-kit to your local machine and launch the notebook with a devoted software (jupyter notebook, jupyter lab, visual studio, ...) from within the starting-kit folder.

The best practice is to create a new environment, e.g., with conda:

`create -n flair-one-baseline python=3.9`

The following libraries are needed (versions indicated ensure a working environment): 

	python==3.9.0
	matplotlib==3.5.2
	scikit-image==0.19.3
	pillow==9.2.0
	torch==1.12.1
	torchmetrics==0.10.0
	pytorch_lightning==1.7.7
	segmentation_models_pytorch==0.3.0
	albumentations==1.2.1
	rasterio==1.2.10
	tensorboard==2.10.1



### Option 2: Google Colab

If you choose to use the notebook from the starting-kit in Google Colab, some steps are needed:

- Create a link of the flair-one-starting-kit shared directory to your drive (right click and "create link in Drive").
Alternatively, you can download and upload the whole content of the starting-kit into your drive.

- Open the flair-one-baseline.ipynb notebook in Colab
- If you are using a link to the shared notebook it has read only rights: select File --> Save a copy in Drive (will make a copy in your drive allowing read and write)
- Select Runtime --> Change runtime type and select GPU
- Uncomment the first notebook cell and run it (check for the path if you are using a copy of the content). 
This will mount your drive to the local Colab VM and allow accessing the dataset files. 
The cell will also install missing libraries on the VM needed to run the baseline code.  



## Reference
Please include a citation to the following paper if you use the FLAIR #1 dataset:

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


## Contact

For any requests, questions, suggestions, feel free to contact us at ai-challenge@ign.fr
