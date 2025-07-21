from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

from mpi4py import MPI
import dill as pickle
import os
import numpy as np
from tqdm import tqdm
from scipy import interpolate
import h5py

def get_inputs(gridvals):
    tmp = np.meshgrid(*gridvals, indexing='ij')
    inputs = np.empty((tmp[0].size, len(tmp)))
    for i, t in enumerate(tmp):
        inputs[:, i] = t.flatten()
    return inputs

def load_completed_hdf5(filename):
    completed_inds = set()
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as f:
            completed_inds = set(map(int, f.keys()))
    return completed_inds

def assign_job(comm, rank, serialized_model, job_iter, inputs):
    try:
        job_index = next(job_iter)
        comm.send((serialized_model, job_index, inputs[job_index]), dest=rank, tag=1)
        return True
    except StopIteration:
        comm.send(None, dest=rank, tag=0)
        return False

def save_result_hdf5(filename, index, x, res):
    with h5py.File(filename, 'a') as f:
        group = f.create_group(str(index))
        group.create_dataset('input', data=x)

        for key, array in res.items():
            group.create_dataset(key, data=array)

def master(model_func, gridvals, filename, progress_filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    inputs = get_inputs(gridvals)

    serialized_model = pickle.dumps(model_func)

    # Check which calculations are already completed
    completed_inds = load_completed_hdf5(filename)
    if completed_inds:
        print(f'Calculations completed/total: {len(completed_inds)}/{inputs.shape[0]}.')
        if len(completed_inds) < inputs.shape[0]:
            print('Restarting calculations...')
        else:
            print('All calculations completed.')
    else:
        with h5py.File(filename, 'w') as f:
            pass  # Just create the file if it doesn't exist

    # Get inputs that have not yet been computed
    job_indices = [i for i in range(len(inputs)) if i not in completed_inds]
    job_iter = iter(job_indices)

    # Open progress log file for writing
    with open(progress_filename, 'w') as log_file:
        pbar = tqdm(total=len(job_indices), file=log_file, dynamic_ncols=True)
        status = MPI.Status()

        # Assign initial workers
        active_workers = 0
        for rank in range(1, size):
            if assign_job(comm, rank, serialized_model, job_iter, inputs):
                active_workers += 1

        # Continue until all workers are terminated
        while active_workers > 0:

            # Get result form worker
            index, x, res = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            worker_rank = status.Get_source()

            # Save the result
            save_result_hdf5(filename, index, x, res)

            pbar.update(1)
            log_file.flush()

            # Assign a new job to the worker.
            if not assign_job(comm, worker_rank, serialized_model, job_iter, inputs):
                active_workers -= 1

        pbar.close()

def worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    while True:
        # Get inputs from master process
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 0:
            break # Shutdown signal

        # Call the function on the inputs
        serialized_model, index, x = data
        model_func = pickle.loads(serialized_model)
        res = model_func(x)

        # Send the results to the master process
        comm.send((index, x, res), dest=0, tag=2)

def make_grid(model_func, gridvals, filename, progress_filename):
    """
    Run a parallel grid computation using MPI, saving results to an HDF5 file.

    This function distributes computations across available MPI ranks. The master
    process assigns jobs to worker processes, collects results, and writes them to
    an HDF5 file. A separate progress log file tracks computation progress.

    Parameters
    ----------
    model_func : callable
        A function that takes a 1D numpy array of input parameters and returns
        a dictionary of results, where each key corresponds to a quantity (numpy array)
        to be saved.

    gridvals : tuple of 1D numpy arrays
        Defines the parameter grid. Each array in the tuple represents the discrete
        values for one dimension of the parameter space.

    filename : str
        Path to the HDF5 file where computed results will be stored. The file will contain
        groups for each grid point index, each with datasets for the input parameters
        and the model output.

    progress_filename : str
        Path to the text file where progress updates (from the master process) will be logged.

    Notes
    -----
    - This function must be run with an MPI launcher (e.g., `mpiexec -n N python script.py`).
    - The results are saved incrementally, so the computation can be resumed if interrupted.
    - Only rank 0 (master) writes to the output files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Master process
        master(model_func, gridvals, filename, progress_filename)
    else:
        # Worker process
        worker()

class GridInterpolator():
    """
    A class for interpolating data saved from an HDF5 grid of simulation outputs.

    This class reads an HDF5 file containing simulation outputs stored on a parameter grid.
    It provides methods to generate interpolators that can predict values or arrays of
    results at arbitrary points within the grid using `scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the grid data.

    gridvals : tuple of np.ndarray
        The parameter values defining each axis of the input grid. The shape of this tuple
        defines the dimensionality of the grid.

    Attributes
    ----------
    gridvals : tuple of np.ndarray
        The parameter values for each grid dimension.

    gridshape : tuple of int
        The shape of the parameter grid, inferred from the lengths of `gridvals`.

    results : list of dict
        A list of results dictionaries read from the HDF5 file, ordered to match the flattened grid.
    """

    def __init__(self, filename, gridvals):
        """
        Initialize the GridInterpolator by loading data from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file containing the simulation results.

        gridvals : tuple of np.ndarray
            The parameter grid values, used to define the interpolation space.

        Notes
        -----
        The HDF5 file should have groups named by index (e.g., '0', '1', ...), each containing
        a 'result' group with datasets corresponding to the keys of the results dictionary.
        """
        self.gridvals = gridvals
        self.gridshape = tuple([len(a) for a in gridvals])
        self.results = self._read_hdf5(filename)

    def _read_hdf5(self, filename):
        results = []
        with h5py.File(filename, 'r') as f:
            # Sort indices numerically to ensure correct order
            sorted_indices = sorted(f.keys(), key=lambda x: int(x))
            for index in sorted_indices:
                group = f[index]
                res_dict = {key: group[key][()] for key in group if key != 'input'}
                results.append(res_dict)
        return results

    def make_array_interpolator(self, key, logspace=False):
        """
        Create an interpolator for a vector quantity across the parameter grid.

        Parameters
        ----------
        key : str
            The key in the results dictionary for which to create the interpolator.
            The corresponding value must be an array (e.g., a profile vs. altitude).

        logspace : bool, optional
            If True, interpolation is performed in log10-space. This is useful for
            quantities that span many orders of magnitude.

        Returns
        -------
        interp_arr : function
            A function that takes a 2D numpy array of shape (n_points, n_dimensions) as input
            and returns an interpolated array corresponding to the chosen key.

        Example
        -------
        >>> interp = grid.make_array_interpolator('mixing_ratio', logspace=True)
        >>> value = interp([[metallicity, temperature]])
        """
        # Make interpolate for key
        interps = []
        for j in range(len(self.results[0][key])):
            val = np.empty(len(self.results))
            for i,r in enumerate(self.results):
                val[i] = r[key][j]
            if logspace:
                val = np.log10(np.maximum(val,2e-38))
            interps.append(val.reshape(self.gridshape))

        vlist = np.asarray(interps)
        vlist = np.moveaxis(vlist, 0, -1)
        rgi = interpolate.RegularGridInterpolator(self.gridvals, vlist)

        def interp_arr(vals):
            out = rgi(vals)[0]
            if logspace:
                out = 10.0**out
            return out

        return interp_arr

    def make_value_interpolator(self, key, logspace=False):
        """
        Create an interpolator for a scalar quantity across the parameter grid.

        Parameters
        ----------
        key1 : str
            The key in the results dictionary for which to create the scalar interpolator.
            The corresponding value must be a scalar (float or int).

        logspace : bool, optional
            If True, interpolation is performed in log10-space.

        Returns
        -------
        interp_val : function
            A function that takes a 2D numpy array of shape (n_points, n_dimensions) and
            returns interpolated scalar values.

        Example
        -------
        >>> interp = grid.make_value_interpolator('surface_temperature', logspace=False)
        >>> temperature = interp([[metallicity, temperature]])
        """

        val = np.empty(len(self.results))

        for i,r in enumerate(self.results):
            val[i] = r[key]

        if logspace:
            val = np.log10(np.maximum(val,2e-38))
        interp = interpolate.RegularGridInterpolator(self.gridvals, val.reshape(self.gridshape))

        def interp_val(vals):
            out = interp(vals)[0]
            if logspace:
                out = 10.0**out
            return out

        return interp_val
