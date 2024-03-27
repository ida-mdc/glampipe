# GLAM: Generative Lung Architecture Modeling (glamPipe)

## Introduction

GLAM: Generative Lung Architecture Modeling (glamPipe) is a powerful and flexible pipeline designed for scientists and researchers in the field of computational biology and medical imaging. It offers a comprehensive suite of tools for 3D image segmentation, 3D mesh generation (.stl files), and preparation of training sets for diffusion process-based image generation. Whether you're analyzing lung architecture or exploring generative models for biological structures, glamPipe provides the necessary functionalities to advance your research.

## Features

- **3D Image Segmentation:** Utilize state-of-the-art algorithms for precise segmentation of 3D images.
- **3D Mesh Generation:** Convert segmented images into high-quality 3D meshes in STL format, suitable for computational analysis and 3D printing.
- **Diffusion Process Preparation:** Prepare your dataset for training with diffusion process-based generative models, enabling the creation of synthetic biological structures.

## Installation
clone the repository and run the following command in the root directory:
`pip install .`

## Command-Line Arguments

This tool supports various command-line arguments to customize its behavior. Below is a detailed description of each argument:

- `-pout`, `--path-output` **(Required)**: Specifies the output path where result directories will be created or found if segmentation was previously done. This argument is required.

- `-s`, `--is_segment`: Enables the segmentation process. When set, path to original images and the segmentation model must be provided unless using default models.

- `-m`, `--is_mesh`: Enables mesh creation from segmented images.

- `-g`, `--is_prep_for_diffusion`: Prepares the segmented images for diffusion processes.

- `-porg`, `--path-originals`: Path to the original images. Required if `-s` is set for segmentation.

- `-sdd`, `--segmentation-dir-date`: Specifies the date of the segmentation directory to use. Should not be provided when performing new segmentation as it implies segmentation results already exist.

- `-c`, `--condition`: Sets the condition for the images being processed. Valid choices are `emphysema`, `healthy`, or `fibrotic`. Required when segmentation is performed.

- `-tm`, `--threshold-method`: Chooses the thresholding method for segmentation. Options are `triangle`, `otsu`, `li`, `yen`. Default is `triangle`.

- `-psm`, `--path-segmentation-model`: Path to the segmentation model file. Required if `-s` is set for segmentation.

- `-dvs`, `--default-voxel-size`: Specifies the default voxel size as a list of three floats. Default is `[0.0000022935, 0.0000013838, 0.0000013838]`.

- `-dmsp`, `--default-mesh-size-in-pixels`: Specifies the default mesh size in pixels as a list of three integers. Default is `[64, 256, 256]`.

- `-gs`, `--gaussian-sigma`: Specifies the Gaussian sigma for smoothing as a list of three floats. Default is `[1.2, 0.8, 0.8]`.

## License
MIT License