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


def check_args(args):
    if not args.is_segment and not args.is_mesh and not args.is_prep_for_diffusion:
        raise ValueError('At least one of the following must be true: is_segment, is_mesh, is_prep_for_diffusion.')
    if args.is_segment and args.path_originals is None:
        raise ValueError('Path to original images must be provided when segmenting.')
    if args.is_segment and args.segmentation_dir_date is not None:
        raise ValueError('segmentation_dir_date should not be provided when segmenting, '
                         'as it implies segmentation results already exist.')
    if args.is_segment and args.condition is None:
        logging.warning('Condition not provided. Will use all images in the directory.')
    if args.is_segment and args.path_segmentation_model is None:
        raise ValueError('Path to segmentation model must be provided when segmenting.')
    if not args.is_segment and args.condition is not None:
        raise ValueError('Condition should only be provided when segmenting.')
    if not os.path.exists(args.path_output):
        raise ValueError('Output path where result dir will be created (or found if segmentation was done)'
                         ' does not exist.')
    if args.segmentation_dir_date:
        if not os.path.exists(os.path.join(args.path_output, args.segmentation_dir_date)):
            raise ValueError('Segmentation results directory does not exist.')


def get_user_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-pout', '--path-output', required=True)
    parser.add_argument('-s', '--is-segment', action='store_true')
    parser.add_argument('-m', '--is-mesh', action='store_true')
    parser.add_argument('-g', '--is-prep-for-diffusion', action='store_true')
    parser.add_argument('-porg', '--path-originals', default=None)
    parser.add_argument('-sdd', '--segmentation-dir-date', default=None)
    parser.add_argument('-c', '--condition', default=None, choices=['emphysema', 'healthy', 'fibrotic'])
    parser.add_argument('-tm', '--threshold-method', default='triangle', choices=['triangle', 'otsu', 'li', 'yen'])
    parser.add_argument('-psm', '--path-segmentation-model', default=None)
    parser.add_argument('-dvs', '--default-voxel-size',
                        default=[0.0000022935, 0.0000013838, 0.0000013838], type=float, nargs='*')
    parser.add_argument('-dmsp', '--default-mesh-size-in-pixels', default=[64, 256, 256], type=int, nargs='*')
    parser.add_argument('-gs', '--gaussian-sigma', default=[1.2, 0.8, 0.8], type=float, nargs='*')

    arguments = parser.parse_args()
    check_args(arguments)
    return arguments


ARGS = get_user_arguments()
TODAY_STR = datetime.today().strftime('%Y%m%d')
