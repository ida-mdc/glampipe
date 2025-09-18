from pathlib import Path
from shutil import rmtree
import pytest


def test_tiles_and_metadata(gp_cfg):
    rmtree(Path(gp_cfg.OUTPUT_PATH_ORIGINAL), ignore_errors=True)
    from glampipe.create_tiles import create_tiles_and_save_metadata
    create_tiles_and_save_metadata()
    # Expect at least one tile and a CSV in the run directory
    run_dir = Path(gp_cfg.OUTPUT_PATH_ORIGINAL).parent  # .../glampipe_<date>_healthy_triangle
    tiles = list(Path(gp_cfg.OUTPUT_PATH_ORIGINAL).glob("*.tif"))
    assert len(tiles) >= 1
    csv = run_dir / "image_properties.csv"
    if not csv.exists():
        csv = Path(gp_cfg.OUTPUT_PATH_ORIGINAL) / "image_properties.csv"
    assert csv.exists()
