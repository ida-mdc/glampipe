import argparse
import logging
import warnings
import os
from datetime import datetime


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_bioimageio = logging.getLogger('bioimageio')
    logger_bioimageio.setLevel(logging.ERROR)
    logger_tf = logging.getLogger('tensorflow')
    logger_tf.setLevel(logging.ERROR)
    os.environ["KMP_WARNINGS"] = "FALSE"
    warnings.filterwarnings('ignore')


def check_args():
    if not ARGS.is_interpolate \
            and not ARGS.is_segment \
            and not ARGS.is_process_probability \
            and not ARGS.is_mesh \
            and not ARGS.is_prep_for_diffusion:
        raise ValueError('At least one of the following must be true:'
                         ' is_interpolate, is_segment, is_process_probability, is_mesh, is_prep_for_diffusion.')
    if ARGS.is_interpolate and ARGS.path_originals is None:
        raise ValueError('Path to original images must be provided when segmenting.')
    if ARGS.is_interpolate and ARGS.dir_date is not None:
        raise ValueError('segmentation_dir_date should not be provided when segmenting, '
                         'as it implies segmentation results already exist.')
    if ARGS.is_interpolate and ARGS.condition is None:
        logging.warning('Condition not provided. Will use all images in the directory.')
    # if ARGS.is_segment and ARGS.path_segmentation_model is None:
    #     raise ValueError('Path to segmentation model must be provided when segmenting.')
    if not os.path.exists(ARGS.output_base_path):
        raise ValueError('Output path where result dir will be created (or found if segmentation was done)'
                         ' does not exist.')
    if ARGS.dir_date:
        if not os.path.exists(OUTPUT_PATH):
            raise ValueError('Segmentation results base directory does not exist.')
        if not os.path.exists(os.path.join(OUTPUT_PATH_INTERPOLATED)):
            raise ValueError('Interpolated images directory does not exist.')


def save_args():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    args = vars(ARGS)
    # create variable filename as a string of current date and time:
    filename = f'args_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    with open(os.path.join(OUTPUT_PATH, filename), 'w') as f:
        for key, value in args.items():
            f.write(f'{key}: {value}\n')


def get_user_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-obp', '--output-base-path', required=True)
    parser.add_argument('-i', '--is-interpolate', action='store_true')
    parser.add_argument('-s', '--is-segment', action='store_true')
    parser.add_argument('-pp', '--is-process-probability', action='store_true')
    parser.add_argument('-m', '--is-mesh', action='store_true')
    parser.add_argument('-pd', '--is-prep-for-diffusion', action='store_true')
    parser.add_argument('-porg', '--path-originals', default=None)
    parser.add_argument('-dd', '--dir-date', default=None)
    parser.add_argument('-c', '--condition', default=None, choices=['emphysema', 'healthy', 'fibrotic'])
    parser.add_argument('-tm', '--threshold-method', default='triangle', choices=['triangle', 'otsu', 'li', 'yen'])
    parser.add_argument('-psm', '--path-segmentation-model', default=None)
    parser.add_argument('-dvs', '--default-voxel-size',
                        default=[0.0000022935, 0.0000013838, 0.0000013838], type=float, nargs='*')
    parser.add_argument('-dmsp', '--default-mesh-size-in-pixels', default=[87, 145, 145], type=int, nargs='*')
    parser.add_argument('-ec', '--is-enhance-contrast', action='store_true')
    parser.add_argument('-gs', '--gaussian-sigma', default=[1.2, 0.8, 0.8], type=float, nargs='*')
    parser.add_argument('-hpt', '--histogram-peak-threshold', default=35, type=int)
    parser.add_argument('-ifs', '--image-final-shape', default=[92, 92, 92], type=int, nargs='*')

    arguments = parser.parse_args()
    return arguments


ARGS = get_user_arguments()

if ARGS.dir_date:
    output_date = ARGS.dir_date
else:
    output_date = datetime.today().strftime('%Y%m%d')

OUTPUT_PATH = os.path.join(ARGS.output_base_path, f'glampipe_{output_date}_{ARGS.condition}_{ARGS.threshold_method}')
OUTPUT_PATH_ORIGINAL = os.path.join(OUTPUT_PATH, 'original')
OUTPUT_PATH_INTERPOLATED = os.path.join(OUTPUT_PATH, 'interpolated')
OUTPUT_PATH_PROBABILITY = os.path.join(OUTPUT_PATH, 'probability')
OUTPUT_PATH_BINARY = os.path.join(OUTPUT_PATH, 'binary')
OUTPUT_PATH_MESH = os.path.join(OUTPUT_PATH, 'meshes')
OUTPUT_PATH_TRAINING_SET = os.path.join(OUTPUT_PATH, 'training_set')
PROPERTIES_FILE = os.path.join(OUTPUT_PATH, 'image_properties.csv')

PLANTSEG_PATH = os.path.join(os.path.expanduser('~'), "miniconda3/envs/glampipe/bin/plantseg")
PLANTSEG_CONFIG_PACKAGE_PATH = os.path.join('glampipe', 'plantset_config.yaml')
PLANTSEG_CONFIG_PATH = os.path.join(OUTPUT_PATH, 'plantset_config.yaml')

check_args()
set_logger()
save_args()
