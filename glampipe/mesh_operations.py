import numpy as np
import trimesh
from scipy.ndimage import binary_dilation
from skimage.measure import marching_cubes
import logging
from glampipe import io_tools
from glampipe.config import OUTPUT_PATH_PROBABILITY


def make_mesh(im, thr, mask):
    im = np.pad(im, pad_width=5, mode='constant', constant_values=0)
    mask = np.pad(mask, pad_width=5, mode='constant', constant_values=0)
    mask = binary_dilation(mask, iterations=3).astype(mask.dtype)

    vertices, faces, _, _ = marching_cubes(im, thr, allow_degenerate=False, mask=mask)

    logging.info('Done: Mesh creation.')

    return vertices, faces


def post_process_mesh(vertices, faces):
    # Convert to a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Remove unnecessary vertices
    mesh.update_faces(mesh.unique_faces())
    non_degenerate_faces = mesh.nondegenerate_faces()
    mesh.update_faces(non_degenerate_faces)
    mesh.process()

    mesh.fix_normals()

    if not mesh.is_watertight:
        mesh.fill_holes()

    logging.info(f'Done: post-processed mesh. - number of faces - {int(len(mesh.faces))}')

    return mesh


def run_mesh_creation():
    paths = io_tools.get_probability_image_paths(OUTPUT_PATH_PROBABILITY)

    for i_p, p in enumerate(paths):

        filename = io_tools.get_filename(p, is_extension=False)

        im = io_tools.read_image(p)
        binary, thr = io_tools.get_binary_image(filename)

        vertices, faces = make_mesh(im, thr, binary)
        mesh = post_process_mesh(vertices, faces)

        io_tools.save_mesh(mesh, filename)
