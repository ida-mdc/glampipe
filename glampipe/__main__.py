from glampipe import segmentation
from glampipe import mesh_operations
from glampipe import prep_diffusion
from glampipe import config  # run config.py
from glampipe.config import ARGS


def main():

    if not ARGS.is_segmentation:
        segmentation.setup_and_run_segmentation()

    if ARGS.is_mesh:
        mesh_operations.run_mesh_creation()

    if ARGS.is_prep_for_diffusion:
        prep_diffusion.run_create_training_set()


if __name__ == "__main__":
    main()
