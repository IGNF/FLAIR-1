#####################################
 FLAIR-one baseline : starting-kit 
#####################################


The starting-kit contains :


|_ toy_dataset_flair-one (folder): 
		sample of the full dataset with same structure and naming convention.

|_ metadata (folder): 
		contains a .json file with the metadata associated to provided patches.

|_ py_module (folder): 
		contains .py files defining the modules used in the notebook.

|_ flair-one-baseline.ipynb (notebook): 
		a notebook allowing to launch the py_module, explore the data and train the baseline.





-------------------------------
!!  flair-one-baseline.ipynb !!  : You can use this notebook in two different ways. Carefully read the following: 
-------------------------------


## option 1 : locally

This option is more practical as the data is relatively volumineous. 
To do so, you just need to download the whole content of the starting-kit to your local machine and launch the notebook with a devoted software (jupyter notebook, jupyter lab, visual studio, ...) from within the starting-kit folder.

::  best practice is to create a new environment, e.g., with conda create -n flair-one-baseline python=3.9
::  the following libraries are needed (versions indicated ensure a working environment): 

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



## option 2 : Google Colab

If you choose to use the notebook from the starting-kit in Google Colab, some steps are needed:

:: create a link of the flair-one-starting-kit shared directory to your drive (right click and "create link in Drive").
Alternatively, you can download and upload the whole content of the starting-kit into your drive.

:: open the flair-one-baseline.ipynb notebook in Colab
:: if you are using a link to the shared notebook it has read only rights: select File --> Save a copy in Drive (will make a copy in your drive allowing read and write)
:: select Runtime --> Change runtime type and select GPU
:: uncomment the first notebook cell and run it (check for the path if you are using a copy of the content). 
This will mount your drive to the local Colab VM and allow accessing the dataset files. 
The cell will also install missing libraries on the VM needed to run the baseline code.  



