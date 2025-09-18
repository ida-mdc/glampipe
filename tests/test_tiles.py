from pathlib import Path
import pytest

def test_tiles_and_metadata(gp_cfg, clean_run_dir):
    from glampipe.create_tiles import create_tiles_and_save_metadata
    create_tiles_and_save_metadata()

    # Run directory like .../glampipe_<date>_healthy_<method>
    run_dir = Path(gp_cfg.OUTPUT_PATH_ORIGINAL).parent
    assert run_dir.exists()

    # CSV should exist either at run root or inside any subdir (some impls write it differently)
    csv = run_dir / "image_properties.csv"
    if not csv.exists():
        # fall back: search recursively
        matches = list(run_dir.rglob("image_properties.csv"))
        assert matches, f"No image_properties.csv under {run_dir}"
        csv = matches[0]
    assert csv.exists()
