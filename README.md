# GLAM: Generative Lung Architecture Modeling (glamPipe)

## Introduction

GLAM: Generative Lung Architecture Modeling (glamPipe) is a flexible pipeline designed for microscopy and bio-medical images. 
It offers a suite of tools for 3D image segmentation, 3D mesh generation (.stl files),  and preparation of training sets for diffusion process-based image generation.  
In combination with training and predicting of the diffusion model, the pipeline supports computationally generating 3D meshes that can be bio-printed to enlarge the imaged dataset for drug and treatment testing.  
Diffusion model: https://github.com/ida-mdc/diff3d 

## Features

- **3D Image Segmentation:** Utilize pre-trained model for precise segmentation of 3D images of tissue - done on printable size patches of the input images.
- **3D Mesh Generation:** Convert segmented images into high-quality 3D meshes in STL format, suitable for 3D printing.
- **Diffusion Process Preparation:** Prepare your dataset for training with diffusion process-based generative models, enabling the creation of synthetic biological structures.

## Installation
Create a conda/virtual env and activate it: 
conda create -n glamPipe python
`conda create -n glamPipe python=3.11`
clone the repository and run the following command in its root directory:  
`pip install .`

## Command-Line Arguments

This tool supports various command-line arguments to customize its behavior. Below is a detailed description of each argument:

- `-obp`, `--output-base-path` **(Required)**: Specifies the output path where result directories will be created or found if segmentation was previously done. This argument is required.

- `-s`, `--is_segment`: Enables the segmentation process. When set, path to original images and the segmentation model must be provided.

- `-m`, `--is_mesh`: Enables mesh creation from segmentation results (probability maps).

- `-g`, `--is_prep_for_diffusion`: Prepares the segmented images as a training set for diffusion processes.

- `-porg`, `--path-originals`: Path to the original images. Required if `-s` is set for segmentation.

- `-sdd`, `--segmentation-dir-date`: Specifies the date of the segmentation directory to use. Should not be provided when performing new segmentation as it implies segmentation results already exist.

- `-c`, `--condition`: Sets the experimental condition for the images being processed. Valid choices are `emphysema`, `healthy`, or `fibrotic`. 
If `None` is left all input images will be processed together. Relevant only for segmentation.

- `-tm`, `--threshold-method`: Chooses the thresholding method for segmentation. Options are `triangle`, `otsu`, `li`, `yen`. Default is `triangle`. Relevant for segmentation and mesh creation.

- `-psm`, `--path-segmentation-model`: Path to the segmentation model file. Required if `-s` is set for segmentation.

- `-dvs`, `--default-voxel-size`: Specifies the default voxel size (from the microscopy settings) as a list of three floats - zxy. Default is `[0.0000022935, 0.0000013838, 0.0000013838]`.

- `-dmsp`, `--default-mesh-size-in-pixels`: Specifies the default mesh size in pixels as a list of three integers - zxy. Default is `[64, 256, 256]`.

- `-gs`, `--gaussian-sigma`: Specifies the Gaussian sigma for smoothing as a list of three floats - zxy. Default is `[1.2, 0.8, 0.8]`.

## License
MIT License