from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

import numpy as np
from gridutils import make_grid

# Only stuff below here needs to change

def model(y):
    a, b = y

    c = 0.0
    for i in range(1_000_000):
        c += 1

    return {'a': a/2, 'b': np.array([b*2])}

def get_gridvals():
    a = np.arange(1, 10, 1.0)
    b = np.arange(1, 20, 1.0)
    gridvals = (a, b)
    return gridvals

if __name__ == '__main__':

    # Call the driver
    make_grid(
        model,
        gridvals=get_gridvals(), 
        filename='results.h5', 
        progress_filename='progress.log'
    )