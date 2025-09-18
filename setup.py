# setup.py
from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="glampipe",
    version="0.1.0",
    description="Image analysis pipeline for the GLAM project.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Ella Bahry",
    license="MIT",
    url="https://github.com/ida-mdc/glampipe",
    project_urls={"Issues": "https://github.com/ida-mdc/glampipe/issues"},
    python_requires=">=3.10,<3.13",
    packages=find_packages(),
    package_data={"glampipe": ["plantset_config.yaml"]},
    include_package_data=True,  # safe alongside package_data
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",   # correct package name (not 'skimage')
        "tifffile>=2021.4.8",
        "pandas>=1.3.3",
        "imageio>=2.9.0",
        "trimesh>=3.9.0",
    ],
    extras_require={
        "dev": ["pytest>=7"],
        "bio": ["xarray>=0.19.0", "bioimageio.core>=0.6.0"],
    },
)
