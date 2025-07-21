# Gridmaker

This is a simple Python package for doing a grid of calculations in parallel using MPI, and saving the results to a HDF5 file. Also, the package contains a class to linearly interpolate the results of the grid calculation. For an example see `tests/test.py`, which should be run with `mpiexec -n 4 python test.py`.