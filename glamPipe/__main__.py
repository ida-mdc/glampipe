from glamPipe import segmentation
from glamPipe import mesh_operations
from glamPipe import prep_diffusion
from glamPipe import config  # run config.py
from glamPipe.config import ARGS


def main():

    if not ARGS.is_segmentation:
        segmentation.setup_and_run_segmentation()

    if ARGS.is_mesh:
        mesh_operations.run_mesh_creation()

    if ARGS.is_prep_for_diffusion:
        prep_diffusion.run_create_training_set()


if __name__ == "__main__":
    main()
