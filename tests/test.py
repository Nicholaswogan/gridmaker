from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

import numpy as np
from gridmaker import make_grid

def model(y):
    x1, x2, x3 = y

    res = np.empty(100)
    for i in range(len(res)):
        res[i] = np.sin(np.pi * x1 / 2) * np.exp(x2 / 2) + x3 + i

    return {'res': res.astype(np.float32), 'x1': x1+x2+x3}

def get_gridvals():
    x1 = np.linspace(-np.pi, np.pi, 5)
    x2 = np.linspace(0.0, np.pi, 5)
    x3 = np.linspace(0.0, np.pi/2, 5)
    gridvals = (x1, x2, x3)
    return gridvals

if __name__ == '__main__':

    # Call the driver
    make_grid(
        model,
        gridvals=get_gridvals(), 
        filename='results.h5', 
        progress_filename='progress.log'
    )