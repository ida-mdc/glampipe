# import bioimageio.core
# from bioimageio.core.resource_tests import test_model
# import numpy as np
# import xarray as xr
# import logging
# from glampipe import io_tools
# from glampipe import image_properties
# from glampipe import image_operations
# from glampipe.config import ARGS, OUTPUT_PATH_PROBABILITY, OUTPUT_PATH_INTERPOLATED
#
#
# def output_model_info(model_resource):
#     logging.info(f'Available weight formats for this model: {model_resource.weights.keys()}')
#     logging.info("The model requires as inputs:")
#     for inp in model_resource.inputs:
#         logging.info(f'Input with axes: {inp.axes} and shape {inp.shape}')
#     logging.info("The model returns the following outputs:")
#     for out in model_resource.outputs:
#         logging.info(f'Output with axes: {out.axes} and shape {out.shape}')
#
#
# def output_model_tests(model_resource):
#     test_result = test_model(model_resource)[0]
#
#     if test_result["status"] == "failed":
#         logging.warning(f'model test: {test_result["name"]}')
#         logging.warning(f'The model test failed with: {test_result["error"]}')
#         logging.warning("with the traceback:")
#         logging.warning("".join(test_result["traceback"]))
#     else:
#         logging.info("The model passed all tests")
#
#
# def load_model(rdf_path):
#     model_resource = bioimageio.core.load_resource_description(rdf_path)
#
#     output_model_info(model_resource)
#     # output_model_tests(model_resource) # only works with internet connection
#
#     prediction_pipeline = bioimageio.core.create_prediction_pipeline(model_resource, devices=None, weight_format=None)
#
#     return model_resource, prediction_pipeline
#
#
# def predict(im, model_resource, prediction_pipeline):
#     input_image = np.transpose(im, (1, 2, 0))[np.newaxis, ..., np.newaxis]
#     input_array = xr.DataArray(input_image, dims=tuple(model_resource.inputs[0].axes))
#
#     prediction = bioimageio.core.predict_with_tiling(prediction_pipeline, input_array, tiling=True, verbose=True)
#
#     output_im = np.transpose(prediction[0][0, :, :, :, 0], [2, 0, 1]).to_numpy()
#
#     logging.info('Done: Segmented image.')
#
#     return output_im
#
#
# def setup_and_run_segmentation():
#     model_resource, prediction_pipeline = load_model(ARGS.path_segmentation_model)
#
#     io_tools.make_output_sub_dir(OUTPUT_PATH_PROBABILITY)
#     paths = io_tools.get_original_image_paths(OUTPUT_PATH_INTERPOLATED, ARGS.condition)
#
#     for i_path, p in enumerate(paths):
#
#         filename = io_tools.get_filename(p)
#         image = io_tools.read_image(p)
#
#         probability_map = predict(image, model_resource, prediction_pipeline)
#
#         is_unusable = image_properties.is_image_too_empty(probability_map)
#
#         if not is_unusable:
#             io_tools.save_image(OUTPUT_PATH_PROBABILITY, filename, probability_map)
#
#         thr = image_properties.get_threshold(probability_map, ARGS.threshold_method)
#         largest_object_mask = image_operations.create_binary(probability_map, thr)
#
#         is_unusable = image_properties.is_image_too_full(largest_object_mask) or \
#                       image_properties.is_image_too_empty(largest_object_mask, 0.1)
#
#         if not is_unusable:
#             io_tools.save_image(OUTPUT_PATH_PROBABILITY, f'{filename[:-4]}_thr{thr}.tif', largest_object_mask)
