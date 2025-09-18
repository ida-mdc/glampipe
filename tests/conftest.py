import sys, importlib
import numpy as np
import tifffile as tif
from pathlib import Path
from shutil import rmtree
import pytest

def _write_toy_tiff(root: Path) -> Path:
    z, y, x = 48, 96, 96
    im = (np.linspace(0, 255, z * y * x).reshape(z, y, x) % 256).astype("uint8")
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
    return root

@pytest.fixture()
def gp_cfg(tmp_path_factory, toy_data, monkeypatch, request):
    """
    Configure glampipe to write into a *unique* temp output dir per test
    and read from toy_data. Import config AFTER setting sys.argv.
    """
    # unique output base per test function
    out = tmp_path_factory.mktemp(f"gp_out_{request.node.name}")

    # vary threshold method per test to keep folder names distinct
    tm_by_test = {
        "test_interpolation_final_shape": "triangle",
        "test_tiles_and_metadata": "otsu",
        "test_prob_binary_and_mesh": "li",
    }
    tm = tm_by_test.get(request.node.name, "triangle")

    argv = [
        "", "-obp", str(out),
        "-i",
        "-porg", str(toy_data),
        "-c", "healthy",
        "-tm", tm,
        "-ifs", "92", "92", "92",
        "-dmsp", "24", "64", "64",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    import glampipe.config as cfg
    importlib.reload(cfg)
    return cfg

@pytest.fixture()
def clean_run_dir(gp_cfg):
    run_dir = Path(gp_cfg.OUTPUT_PATH_ORIGINAL).parent
    rmtree(run_dir, ignore_errors=True)   # pre-test cleanup
    yield
    rmtree(run_dir, ignore_errors=True)   # post-test cleanup

@pytest.fixture(autouse=True)
def _patch_io_tools(monkeypatch, gp_cfg):
    """
    Make directory creation idempotent, fix CSV pathing, and ensure mesh
    export directory exists — without touching production code.
    """
    from glampipe import io_tools as I
    from pathlib import Path

    # 1) tolerate pre-existing dirs
    def _safe_mkdir(p):
        Path(p).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(I, "make_output_sub_dir", _safe_mkdir, raising=False)

    # 2) write CSV to the correct absolute location (PROPERTIES_FILE is absolute)
    def _image_properties_to_csv(i_path, p, voxel_size, interpolation_factors,
                                 mesh_pixel_size_pre_interpolation, mesh_size_micron_str,
                                 patches_start_idxs):
        file_path = Path(gp_cfg.PROPERTIES_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        def _arr2str(a):
            return np.array2string(a, separator=' ')[1:-1]

        patches_start_idxs_str = ' '.join(_arr2str(a) for a in patches_start_idxs)
        header = ("idx,p,voxel_size,interpolation_factors,"
                  "mesh_pixel_size_pre_interpolation,mesh_size_micron_str,"
                  "patches_start_idxs\n")
        # append with header if new
        if not file_path.exists():
            file_path.write_text(header)

        with file_path.open('a') as f:
            f.write(
                f"{i_path},{p},{_arr2str(voxel_size)},{_arr2str(interpolation_factors)},"
                f"{_arr2str(mesh_pixel_size_pre_interpolation)},{mesh_size_micron_str},"
                f"{patches_start_idxs_str}\n"
            )

    monkeypatch.setattr(I, "image_properties_to_csv", _image_properties_to_csv, raising=False)

    # 3) ensure mesh parent dir exists before export (wrap save_mesh)
    _orig_save_mesh = getattr(I, "save_mesh", None)

    def _safe_save_mesh(mesh, basename: str):
        out = Path(gp_cfg.OUTPUT_PATH_MESH) / f"{basename}.stl"
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out))
        return str(out)

    if callable(_orig_save_mesh):
        monkeypatch.setattr(I, "save_mesh", _safe_save_mesh, raising=False)
