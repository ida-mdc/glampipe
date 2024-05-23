import logging
from glampipe import image_properties
from glampipe import image_operations
from glampipe import io_tools
from glampipe.config import OUTPUT_PATH_PROBABILITY_PROCESSED, ARGS


def run_create_training_set():

    paths = io_tools.get_probability_image_paths(OUTPUT_PATH_PROBABILITY_PROCESSED)
    for i_p, p in enumerate(paths):

        filename = io_tools.get_filename(p)
        im = io_tools.read_image(p)

        if not image_properties.is_histogram_peak_above_threshold(im):
            logging.info(f'Image {filename} histogram peak below threshold. Skipping')
            continue

        resize_factors = [nn_dim / im_dim for nn_dim, im_dim in zip(ARGS.image_shape_neural_network, im.shape)]
        resized_im = image_operations.resize_interpolate_image(im, resize_factors, 1)

        io_tools.save_training_set_image(filename, resized_im)
        # io_tools.save_as_gif(resized_im, filename)
