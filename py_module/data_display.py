#!/usr/bin/env python
# coding: utf-8


## Imports
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot




lut_colors = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

lut_classes = {
1   : 'building',
2   : 'pervious surface',
3   : 'impervious surface',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous vegetation',
11  : 'agricultural land',
12  : 'plowed land',
13  : 'swimming_pool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}

## Functions

def get_data_paths (path, filter):
    for path in Path(path).rglob(filter):
         yield path.resolve().as_posix()


def remapping(lut: dict, recover='color') -> dict:
    rem = lut.copy()
    for idx in [13,14,15,16,17,18,19]: del rem[idx]
    if recover == 'color':  rem[13] = '#000000'
    elif recover == 'class':  rem[13] = 'other'
    return rem


def convert_to_color(arr_2d: np.ndarray, palette: dict = lut_colors) -> np.ndarray:
    rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in rgb_palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def display_nomenclature() -> None:   
    GS = matplotlib.gridspec.GridSpec(1,2)
    fig = plt.figure(figsize=(15,10))
    fig.patch.set_facecolor('black')

    plt.figtext(0.73,0.92, "REDUCED (BASELINE) NOMENCLATURE", ha="center", va="top", fontsize=14, color="w")
    plt.figtext(0.3, 0.92, "FULL NOMENCLATURE", ha="center", va="top", fontsize=14, color="w")

    full_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[0])
    for u,k in enumerate(lut_classes):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=full_nom[u], width_ratios=[2,6])
        ax_color, ax_class = fig.add_subplot(curr_gs[0], xticks=[], yticks=[]), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(lut_colors[k])
        ax_class.text(0.05,0.3, f'({u+1}) - '+lut_classes[k], fontsize=14, fontweight='bold')
    main_nom = matplotlib.gridspec.GridSpecFromSubplotSpec(19, 1, subplot_spec=GS[1])
    for u,k in enumerate(remapping(lut_classes, recover='class')):
        curr_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_nom[u], width_ratios=[2,6])
        ax_color, ax_class = fig.add_subplot(curr_gs[0], xticks=[], yticks=[]), fig.add_subplot(curr_gs[1], xticks=[], yticks=[])
        ax_color.set_facecolor(remapping(lut_colors, recover='color')[k])
        ax_class.text(0.05,0.3, f'({k}) - '+(remapping(lut_classes, recover='class')[k]), fontsize=14, fontweight='bold')
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('w'), spine.set_linewidth(1.5)
    plt.show()    

def display_samples(images, masks, nb_samples: list, palette=lut_colors) -> None:
    indices= random.sample(range(0, len(images)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 3, figsize = (20, nb_samples * 6)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
    for u, idx in enumerate(indices):
        with rasterio.open(images[idx], 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(masks[idx], 'r') as f:
            mk = f.read([1])
            mk = convert_to_color(mk[0], palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(mk, interpolation='nearest') ;ax1.axis('off')
        ax2 = axs[u][2] ; ax2.imshow(im); ax2.imshow(mk, interpolation='nearest', alpha=0.25); ax2.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('Ground Truth Mask', size=16,fontweight="bold",c='w')
            ax2.set_title('Overlay Image & Mask', size=16,fontweight="bold",c='w')    
                 
def display_all(images, masks) -> None:
    GS = matplotlib.gridspec.GridSpec(25,10, wspace=0.002, hspace=0.1)
    fig = plt.figure(figsize=(40,100))
    fig.patch.set_facecolor('black')
    for u,k in enumerate(images):
        ax=fig.add_subplot(GS[u], xticks=[], yticks=[])
        with rasterio.open(k, 'r') as f:
            img = f.read([1,2,3])
        rasterio.plot.show(img, ax=ax)
        ax.set_title(k.split('/')[-1][:-4], color='w')
        get_m = [i for i in masks if k.split('/')[-1].split('_')[1][:-4] in i][0]
        with rasterio.open(get_m, 'r') as f:
            msk = f.read()        
        ax.imshow(convert_to_color(msk[0], palette=lut_colors), interpolation='nearest', alpha=0.2)
    plt.show()
    
    
def display_all_with_semantic_class(images, masks: list, semantic_class: int) -> None:
    
    def convert_to_color_and_mask(arr_2d: np.ndarray, semantic_class: int, palette: dict = lut_colors) -> np.ndarray:
        rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
        arr_3d = np.zeros((arr_2d[0].shape[0], arr_2d[0].shape[1], 4), dtype=np.uint8)
        for c, i in rgb_palette.items():
            m = arr_2d[0] == c
            if c == semantic_class:
                g = list(i)
                g.append(150)
                u = tuple(g)
                arr_3d[m] = u
            else:
                arr_3d[m] = tuple([0,0,0,0])   
        return arr_3d  
    
    sel_imgs, sel_msks, sel_ids = [],[],[]
    for img,msk in zip(images, masks):
        with rasterio.open(msk, 'r') as f:
            data_msk = f.read()
        if semantic_class in list(set(data_msk.flatten())):
            sel_msks.append(convert_to_color_and_mask(data_msk, semantic_class, palette=lut_colors))
            with rasterio.open(img, 'r') as f:
                data_img = f.read([1,2,3])
            sel_imgs.append(data_img)
            sel_ids.append(img.split('/')[-1][:-4]) 
    if len(sel_imgs) == 0:
        print('='*50, f'      SEMANTIC CLASS: {lut_classes[semantic_class]}', '...CONTAINS NO IMAGES IN THE CURRENT DATASET!...',  '='*50, sep='\n')        
    else:
        print('='*50, f'      SEMANTIC CLASS: {lut_classes[semantic_class]}', '='*50, sep='\n')    
        GS = matplotlib.gridspec.GridSpec(int(np.ceil(len(sel_imgs)/5)),5, wspace=0.002, hspace=0.1)
        fig = plt.figure(figsize=(30,6*int(np.ceil(len(sel_imgs)/5))))
        fig.patch.set_facecolor('black')
        for u, (im,mk,na) in enumerate(zip(sel_imgs, sel_msks, sel_ids)):
            ax=fig.add_subplot(GS[u], xticks=[], yticks=[])
            ax.set_title(na, color='w')
            ax.imshow(im.swapaxes(0, 2).swapaxes(0, 1))       
            ax.imshow(mk, interpolation='nearest')
        plt.show()



def display_predictions(images, predictions, nb_samples: int, palette=lut_colors, classes=lut_classes) -> None:
    indices= random.sample(range(0, len(predictions)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 2, figsize = (17, nb_samples * 8)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
  
    palette = remapping(palette, recover='color')
    classes = remapping(classes, recover='class')

    for u, idx in enumerate(indices):
        rgb_image = [i for i in images if predictions[idx].split('_')[-1][:-4] in i][0]
        with rasterio.open(rgb_image, 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(predictions[idx], 'r') as f:
            mk = f.read([1])+1
            f_classes = np.array(list(set(mk.flatten())))
            mk = convert_to_color(mk[0], palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(mk, interpolation='nearest', alpha=1); ax1.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('Prediction', size=16,fontweight="bold",c='w')
        handles = []
        for val in f_classes:
            handles.append(mpatches.Patch(color=palette[val], label=classes[val]))
        leg = ax1.legend(handles=handles, ncol=1, bbox_to_anchor=(1.4,1.01), fontsize=12, facecolor='k') 
        for txt in leg.get_texts():
          txt.set_color('w')
