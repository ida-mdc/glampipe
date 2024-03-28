import logging
import os
from glamPipe.config import ARGS, set_logger
from glamPipe import io_tools
from glamPipe import image_properties
from glamPipe import segmentation
from glamPipe import mesh_operations
from glamPipe import image_operations


def main():
    set_logger()

    if not ARGS.is_segmentation:
        segmentation.setup_and_run_segmentation()
    else:
        output_dir_path = io_tools.get_existing_output_dir()

            # vertices, faces = mesh_operations.make_mesh(probability_map_upsampled, thr, largest_object_mask)
            # mesh = mesh_operations.post_process_mesh(vertices, faces)
            #
            # io_tools.save_images_and_mesh(output_dir_path,
            #                               i_path,
            #                               i_patch,
            #                               patch,
            #                               probability_map,
            #                               largest_object_mask,
            #                               mesh,
            #                               mesh_size_micron_str)


if __name__ == "__main__":
    main()
