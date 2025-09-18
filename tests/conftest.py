import os, sys, importlib
import numpy as np
import tifffile as tif
import pytest

def _write_toy_tiff(root):
    z, y, x = 48, 96, 96
    im = (np.linspace(0, 255, z*y*x).reshape(z, y, x) % 256).astype("uint8")
    # ImageJ-style "Info" with voxel sizes on Z and X (example values)
    info = (
        "Scaling|Distance|Value #1=0.0000013838\n"
        "Scaling|Distance|Value #3=0.0000022935"
    )
    d = root / "2024_dummy"
    d.mkdir(parents=True, exist_ok=True)
    tif.imwrite(d / "healthy_sample.tif", im, imagej=True, metadata={"Info": info})
    return d

@pytest.fixture(scope="session")
def toy_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("gp_data")
    _write_toy_tiff(root)
    return root  # e.g. /tmp/.../gp_data

@pytest.fixture()
def gp_cfg(tmp_path, toy_data, monkeypatch):
    """
    Configure glampipe to write into a temp output dir and read from toy_data.
    Important: set sys.argv BEFORE importing config, then reload.
    """
    out = tmp_path / "gp_out"
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        "",                  # program
        "-obp", str(out),    # OUTPUT_BASE_PATH
        "-i",                # interpolate stage enabled (if your config uses flags)
        "-porg", str(toy_data),  # parent of year-folder (matches your discovery)
        "-c", "healthy",     # class filter
        "-ifs", "92", "92", "92",  # final shape for interpolation tests
        "-dmsp", "24", "64", "64",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # stability
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    import glampipe.config as cfg
    importlib.reload(cfg)
    return cfg
