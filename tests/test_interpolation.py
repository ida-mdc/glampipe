from tifffile import imread
from pathlib import Path
import pytest

def test_interpolation_final_shape(gp_cfg):

    # Import AFTER gp_cfg has set sys.argv & reloaded glampipe.config
    from glampipe.create_tiles import create_tiles_and_save_metadata
    from glampipe.interpolate import run_interpolation

    # Ensure tiles exist if running this test in isolation
    if not list(Path(gp_cfg.OUTPUT_PATH_ORIGINAL).glob("*.tif")):
        create_tiles_and_save_metadata()

    run_interpolation()

    outs = list(Path(gp_cfg.OUTPUT_PATH_INTERPOLATED).glob("*.tif"))
    assert outs, "no interpolated tiles"
    im = imread(outs[0])

    expected = tuple(int(x) for x in gp_cfg.ARGS.image_final_shape)
    assert tuple(im.shape) == expected
