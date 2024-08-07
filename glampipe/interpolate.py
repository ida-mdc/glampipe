from glampipe.config import OUTPUT_PATH_ORIGINAL, OUTPUT_PATH_INTERPOLATED, ARGS
from glampipe import io_tools
from glampipe import image_operations


def run_interpolation():
    io_tools.make_output_sub_dir(OUTPUT_PATH_INTERPOLATED)
    paths = io_tools.get_image_paths(OUTPUT_PATH_ORIGINAL)

    for i_p, p in enumerate(paths):

        filename = io_tools.get_filename(p)
        interpolation_factors = io_tools.get_interpolation_factors_from_csv(int(filename.split('_')[0]))

        image = io_tools.read_image(p)

        image_resampled = image_operations.resize_interpolate_image(image, interpolation_factors)

        resize_factors = [nn_dim / im_dim for nn_dim, im_dim in zip(ARGS.image_final_shape, image_resampled.shape)]
        resized_im = image_operations.resize_interpolate_image(image_resampled, resize_factors, 1)

        io_tools.save_image(OUTPUT_PATH_INTERPOLATED, filename, resized_im)
