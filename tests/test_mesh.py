# tests/test_mesh.py
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter

def test_prob_binary_and_mesh(gp_cfg):
    from glampipe.create_tiles import create_tiles_and_save_metadata
    from glampipe.interpolate import run_interpolation
    from glampipe.image_operations import create_binary
    from glampipe.mesh_operations import make_mesh, post_process_mesh
    from glampipe.io_tools import save_mesh

    # Ensure inputs exist
    if not list(Path(gp_cfg.OUTPUT_PATH_ORIGINAL).glob("*.tif")):
        create_tiles_and_save_metadata()
    if not list(Path(gp_cfg.OUTPUT_PATH_INTERPOLATED).glob("*.tif")):
        run_interpolation()

    outs = list(Path(gp_cfg.OUTPUT_PATH_INTERPOLATED).glob("*.tif"))
    if outs:
        vol = imread(outs[0]).astype("float32")
    else:
        # fallback: synthetic blob
        z = y = x = 64
        zz, yy, xx = np.mgrid[:z, :y, :x]
        c = np.array([z/2, y/2, x/2])
        r = 20.0
        dist2 = (zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2
        vol = (np.exp(-dist2 / (2*(r/2)**2)) * 255.0).astype("float32")

    # prob → binary
    prob = (gaussian_filter(vol, sigma=1) / max(1.0, float(vol.max()))) * 255.0
    thr = 80.0
    bin_ = create_binary(prob, thr)

    # ensure output dirs exist
    Path(gp_cfg.OUTPUT_PATH_PROBABILITY).mkdir(parents=True, exist_ok=True)
    Path(gp_cfg.OUTPUT_PATH_BINARY).mkdir(parents=True, exist_ok=True)
    Path(gp_cfg.OUTPUT_PATH_MESH).mkdir(parents=True, exist_ok=True)

    # save intermediates (optional)
    imwrite(Path(gp_cfg.OUTPUT_PATH_PROBABILITY) / "toy_prob_thr80.tif", prob.astype("float32"))
    imwrite(Path(gp_cfg.OUTPUT_PATH_BINARY) / "toy_prob_thr80.tif", bin_.astype("uint8"))

    # mesh
    verts, faces = make_mesh(prob, thr, bin_)
    mesh = post_process_mesh(verts, faces)
    save_mesh(mesh, "toy_prob_thr80")

    stl = Path(gp_cfg.OUTPUT_PATH_MESH) / "toy_prob_thr80.stl"
    assert stl.exists() and stl.stat().st_size > 0
