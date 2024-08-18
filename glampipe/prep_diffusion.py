from glampipe import io_tools
from glampipe.config import OUTPUT_PATH_TRAINING_SET
import torchio as tio
from torch.utils.data import DataLoader
import numpy as np
import secrets
import os
import tifffile as tif
import random


def random_multiple_of_90():
    deg = random.choice([0, 90, 180, 270])
    return deg, deg


def run_create_training_set():

    io_tools.make_output_sub_dir(OUTPUT_PATH_TRAINING_SET)
    paths = io_tools.get_probability_image_paths()

    for epoch in range(0, 200):
        random_string = secrets.token_hex(4)

        random_affine = tio.RandomAffine(
            scales=(1, 1),  # Keep the scale unchanged
            degrees=(*random_multiple_of_90(), *random_multiple_of_90(), *random_multiple_of_90()),
            isotropic=True,  # Keep the voxel spacing after rotation
            center='image'  # Rotate around the center of the image
        )

        transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.RandomElasticDeformation(
                num_control_points=(5, 5, 5),
                max_displacement=(4, 4, 4),
                locked_borders=1
            ),
            random_affine,
            tio.RandomFlip(axes=(0, 1, 2)),
        ])

        subjects = []

        for i_p, p in enumerate(paths):

            subject = tio.Subject(
                image=tio.ScalarImage(p),
            )
            subjects.append(subject)

        dataset = tio.SubjectsDataset(subjects, transform=transforms)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for step, batch in enumerate(dataloader):

            im = batch['image']['data'][0, 0].cpu().numpy()
            im = np.moveaxis(im, 2, 0)

            new_filename = f"{os.path.basename(paths[step])[:-5]}_{random_string}.tif"

            tif.imwrite(os.path.join(OUTPUT_PATH_TRAINING_SET, new_filename), im)
