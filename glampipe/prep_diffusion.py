import logging
from glampipe import image_properties
from glampipe import image_operations
from glampipe import io_tools
from glampipe.config import OUTPUT_PATH_PROBABILITY, ARGS


def run_create_training_set():

    paths = io_tools.get_probability_image_paths(OUTPUT_PATH_PROBABILITY)
    for i_p, p in enumerate(paths):

        filename = io_tools.get_filename(p, is_extension=False)
        im = io_tools.read_image(p)

        if not image_properties.is_histogram_peak_above_threshold(im):
            logging.info(f'Image {filename} histogram peak below threshold. Skipping')
            continue
