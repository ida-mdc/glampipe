import bioimageio.core
from bioimageio.core.resource_tests import test_model
import numpy as np
import xarray as xr
import logging
from glampipe import io_tools
from glampipe import image_properties
from glampipe import image_operations
from glampipe.config import ARGS, OUTPUT_PATH_PROBABILITY


def output_model_info(model_resource):
    logging.info(f'Available weight formats for this model: {model_resource.weights.keys()}')
    logging.info("The model requires as inputs:")
    for inp in model_resource.inputs:
        logging.info(f'Input with axes: {inp.axes} and shape {inp.shape}')
    logging.info("The model returns the following outputs:")
    for out in model_resource.outputs:
        logging.info(f'Output with axes: {out.axes} and shape {out.shape}')


def output_model_tests(model_resource):
    test_result = test_model(model_resource)[0]

    if test_result["status"] == "failed":
        logging.warning(f'model test: {test_result["name"]}')
        logging.warning(f'The model test failed with: {test_result["error"]}')
        logging.warning("with the traceback:")
        logging.warning("".join(test_result["traceback"]))
    else:
        logging.info("The model passed all tests")


def load_model(rdf_path):
    model_resource = bioimageio.core.load_resource_description(rdf_path)

    output_model_info(model_resource)
    output_model_tests(model_resource)

    prediction_pipeline = bioimageio.core.create_prediction_pipeline(model_resource, devices=None, weight_format=None)

    return model_resource, prediction_pipeline


def predict(im, model_resource, prediction_pipeline):
    input_image = np.transpose(im, (1, 2, 0))[np.newaxis, ..., np.newaxis]
    input_array = xr.DataArray(input_image, dims=tuple(model_resource.inputs[0].axes))

    prediction = bioimageio.core.predict_with_tiling(prediction_pipeline, input_array, tiling=True, verbose=True)

    output_im = np.transpose(prediction[0][0, :, :, :, 0], [2, 0, 1]).to_numpy()

    logging.info('Done: Segmented image.')

    return output_im


def run_process_probability():

    paths = io_tools.get_probability_image_paths(OUTPUT_PATH_PROBABILITY)

    for i_p, p in enumerate(paths):

        filename = io_tools.get_filename(p)
        interpolation_factors = io_tools.get_interpolation_factors_from_csv(int(filename.split('_')[0]))

        probability_map = io_tools.read_image(p)

        probability_map_smooth = image_operations.smooth_image(probability_map, ARGS.gaussian_sigma)
        probability_map_upsampled = image_operations.interpolate_for_upsample(probability_map_smooth,
                                                                              interpolation_factors)
        thr = image_properties.get_threshold(probability_map_upsampled, ARGS.threshold_method)
        largest_object_mask = image_operations.create_binary(probability_map_upsampled, thr)

        io_tools.save_processed_probability_images(filename, largest_object_mask, probability_map_upsampled, thr)


def setup_and_run_segmentation():
    io_tools.make_output_sub_dirs()
    paths = io_tools.get_original_image_paths(ARGS.path_originals, ARGS.condition)
    model_resource, prediction_pipeline = load_model(ARGS.path_segmentation_model)

    for i_path, p in enumerate(paths):

        logging.info(f'image {i_path}: {p}')

        voxel_size = image_properties.get_voxel_size(p)
        interpolation_factors = image_properties.get_interpolation_factor(voxel_size, ARGS.default_voxel_size)
        mesh_pixel_size_pre_interpolation = image_properties.get_mesh_size_in_pixels_pre_interpolation(
            ARGS.default_mesh_size_in_pixels, interpolation_factors)
        mesh_size_micron_str = image_properties.get_mesh_size_micron_str(mesh_pixel_size_pre_interpolation,
                                                                         voxel_size)

        im = io_tools.read_image(p, mesh_pixel_size_pre_interpolation)
        if isinstance(im, bool):
            continue

        patches_start_idxs = image_properties.get_patches_start_idxs(im.shape, mesh_pixel_size_pre_interpolation)

        io_tools.image_properties_to_csv(i_path, p, voxel_size, interpolation_factors,
                                         mesh_pixel_size_pre_interpolation, mesh_size_micron_str,
                                         patches_start_idxs)

        for i_patch, patch_start_idxs in enumerate(patches_start_idxs):
            patch = image_operations.extract_patch(im, patch_start_idxs, mesh_pixel_size_pre_interpolation)

            probability_map = predict(patch, model_resource, prediction_pipeline)

            io_tools.save_patch_segmentation_images(i_path, i_patch, patch, probability_map)
