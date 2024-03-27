import bioimageio.core
from bioimageio.core.resource_tests import test_model
import numpy as np
import xarray as xr
import logging


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
