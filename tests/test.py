from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

import numpy as np
from gridmaker import make_grid, GridInterpolator
from mpi4py import MPI

def model(y):
    x1, x2, x3 = y

    res = np.empty(100)
    for i in range(len(res)):
        res[i] = np.sin(np.pi * x1 / 2) * np.exp(x2 / 2) + x3 + i

    return {'res': res.astype(np.float32), 'a': x1+x2+x3}

def get_gridvals():
    x1 = np.linspace(-np.pi, np.pi, 5)
    x2 = np.linspace(0.0, np.pi, 5)
    x3 = np.linspace(0.0, np.pi/2, 5)
    gridvals = (x1, x2, x3)
    return gridvals

def test_interpolator():

    g = GridInterpolator('results.h5',get_gridvals())
    interp_res = g.make_interpolator('res')
    interp_a = g.make_interpolator('a')

    y = (-1.57079633, 1.57079633, 1.17809725)
    result = model(y)
    res1 = result['res']
    a1 = result['a']
    res2 = interp_res(y)
    a2 = interp_a(y)

    assert np.allclose(res1, res2)
    assert np.isclose(a1, a2)

if __name__ == '__main__':

    # Call the driver
    make_grid(
        model,
        gridvals=get_gridvals(), 
        filename='results.h5', 
        progress_filename='progress.log'
    )

    # Now try interpolator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        test_interpolator()
    
