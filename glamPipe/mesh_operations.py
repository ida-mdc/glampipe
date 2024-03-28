import numpy as np
import trimesh
from scipy.ndimage import binary_dilation
from skimage.measure import marching_cubes
import os
import logging
from glamPipe import io_tools


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


def save_mesh(mesh, output_dir, i_path, i_patch, mesh_size_micron):
    filepath = os.path.join(output_dir, 'meshes', f'{i_path}_{i_patch}_{mesh_size_micron}micron.stl')
    mesh.export(filepath)


def run_mesh_creation(output_dir_path):
    # This function needs work! Missing funcs.
    paths = io_tools.get_segmenation_file_paths()

    for i_path, path in enumerate(paths):

        sub_str = 'xxx'
        thr = 'xxx'

        im = io_tools.read_image(path)
        binary = io_tools.read_image(path)

        vertices, faces = make_mesh(im, thr, binary)
        mesh = post_process_mesh(vertices, faces)

        io_tools.save_mesh(output_dir_path, sub_str, mesh)
