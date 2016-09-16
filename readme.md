# Training and testing models for end-to-end KITTI odometry

Non-standard dependencies:
Torch packages [matio](https://github.com/soumith/matio-ffi.torch), [hdf5](https://github.com/deepmind/torch-hdf5)
Python packages

A sample script is provided to train a basic model architecture on one KITTI subsequence for one epoch. The resulting model is then used to estimate the translation on a KITTI test subsequence. Errors and the estimated trajectory are plotted. As expected, it performs pretty poorly after a single epoch on this limited data!

To run the sample script:
./sample-train-test.sh

To train a model with the default settings:
th ./train-kitti-net.lua

Command-line options are listed in setup.lua.
