# VBFDML2

New(er) version of (voxel-based approximation of) few distance-measurement localization, with support for uncertain environments. 
This repositoray has a single C++ header which may be included in any project, as well as a pybind11 Python bindings for this library.

## Install

### Prerequisites

For Ubuntu 20.04 LTS (focal):

    sudo apt-get install libfmt-dev libcgal-dev libglm-dev

For macOS (Tested an Apple Sillicon):

    brew install libomp fmt cgal glm

### Python Bindings

To install the Python bindings, run:

    pip3 install ./libs/vbfdml2

## Usage

See `python/demo.py` for an example.