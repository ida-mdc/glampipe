import subprocess
from glampipe.config import OUTPUT_PATH_PROBABILITY, OUTPUT_PATH_INTERPOLATED, PLANTSEG_PATH, PLANTSEG_CONFIG_PATH
from glampipe import io_tools


def setup_and_run_segmentation():

    # io_tools.make_output_sub_dir(OUTPUT_PATH_PROBABILITY)
    # paths = io_tools.get_original_image_paths(OUTPUT_PATH_INTERPOLATED, ARGS.condition)

    io_tools.replace_string_in_file(PLANTSEG_CONFIG_PATH, 'TEMPLATE_PATH', OUTPUT_PATH_INTERPOLATED)

    result = subprocess.run([PLANTSEG_PATH, "--config", PLANTSEG_CONFIG_PATH], capture_output=True, text=True)

    if result.returncode == 0:
        print("Plant-seg command executed successfully!")
        print("Output:", result.stdout)
    else:
        print("Plant-seg Command failed with error:")
        print(result.stderr)

    # for i_path, p in enumerate(paths):
    #
    #     filename = io_tools.get_filename(p)
    #     image = io_tools.read_image(p)
    #
    #     probability_map = predict(image, model_resource, prediction_pipeline)
    #
    #     is_unusable = image_properties.is_image_too_empty(probability_map)
    #
    #     if not is_unusable:
    #         io_tools.save_image(OUTPUT_PATH_PROBABILITY, filename, probability_map)
    #
    #     thr = image_properties.get_threshold(probability_map, ARGS.threshold_method)
    #     largest_object_mask = image_operations.create_binary(probability_map, thr)
    #
    #     is_unusable = image_properties.is_image_too_full(largest_object_mask) or \
    #                   image_properties.is_image_too_empty(largest_object_mask, 0.1)
    #
    #     if not is_unusable:
    #         io_tools.save_image(OUTPUT_PATH_PROBABILITY, f'{filename[:-4]}_thr{thr}.tif', largest_object_mask)
