from setuptools import setup, find_packages

setup(
    name='glamPipe',
    version='0.1.0',
    description='Image analysis pipeline for the GLAM project.',
    author='Ella Bahry',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.7.0',
        'skimage>=0.19.0',
        'tifffile>=2021.4.8',
        'pandas>=1.3.3',
        'imageio>=2.9.0',
        'xarray>=0.19.0',
        'bioimageio>=0.6.0',
        'trimesh>=3.9.0',
    ],
)
