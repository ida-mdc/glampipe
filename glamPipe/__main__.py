from glamPipe.config import ARGS, set_logger
from glamPipe import io_tools
from glamPipe import segmentation
from glamPipe import mesh_operations
from glamPipe import prep_diffusion


def main():
    set_logger()

    if not ARGS.is_segmentation:
        segmentation.setup_and_run_segmentation()
    else:
        output_dir_path = io_tools.get_existing_output_dir()

    if ARGS.is_mesh:
        mesh_operations.run_mesh_creation(output_dir_path)

    if ARGS.is_prep_for_diffusion:
        prep_diffusion.run_create_training_set(output_dir_path)


if __name__ == "__main__":
    main()
