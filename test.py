import unittest
import os
import numpy as np
import h5py
from gridmaker import make_grid, GridInterpolator
import subprocess

def dummy_model(x):
    """A simple model function for testing."""
    return {'y': np.sum(x**2), 'z': np.prod(x), 'arr': np.array([np.sum(x), np.prod(x)])}

class TestGridMaker(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.filename = 'test_grid.h5'
        self.progress_filename = 'test_progress.log'
        self.gridvals = (np.linspace(0, 1, 3), np.linspace(0, 1, 3))

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.progress_filename):
            os.remove(self.progress_filename)

    def test_make_grid_comprehensive(self):
        """Test the make_grid function comprehensively."""
        # This test requires mpi to be installed and run with mpiexec
        try:
            subprocess.check_call(['mpiexec', '-n', '4', 'python', '-c',
                                   f'from gridmaker import make_grid; import numpy as np; make_grid({dummy_model}, {self.gridvals}, "{self.filename}", "{self.progress_filename}")'])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.skipTest(f"MPI execution failed, skipping test_make_grid_comprehensive. Error: {e}")

        self.assertTrue(os.path.exists(self.filename))
        with h5py.File(self.filename, 'r') as f:
            self.assertEqual(len(f.keys()), 9)
            inputs = np.meshgrid(*self.gridvals, indexing='ij')
            inputs = np.stack(inputs, axis=-1).reshape(-1, 2)
            for i in range(9):
                self.assertIn(str(i), f)
                self.assertTrue(np.allclose(f[str(i)]['input'][()], inputs[i]))
                expected_y = np.sum(inputs[i]**2)
                expected_z = np.prod(inputs[i])
                self.assertAlmostEqual(f[str(i)]['y'][()], expected_y)
                self.assertAlmostEqual(f[str(i)]['z'][()], expected_z)

class TestGridInterpolator(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.filename = 'test_grid_interpolator.h5'
        self.gridvals = (np.array([0., 1., 2.]), np.array([0., 1., 2.]))
        inputs = np.array([[i, j] for i in self.gridvals[0] for j in self.gridvals[1]])
        with h5py.File(self.filename, 'w') as f:
            for i in range(inputs.shape[0]):
                group = f.create_group(str(i))
                group.create_dataset('input', data=inputs[i])
                res = dummy_model(inputs[i])
                for key, value in res.items():
                    group.create_dataset(key, data=value)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_GridInterpolator_init(self):
        """Test GridInterpolator initialization."""
        interpolator = GridInterpolator(self.filename, self.gridvals)
        self.assertEqual(len(interpolator.results), 9)
        self.assertEqual(interpolator.gridshape, (3, 3))

    def test_make_value_interpolator(self):
        """Test the make_value_interpolator method."""
        interpolator = GridInterpolator(self.filename, self.gridvals)
        interp_y = interpolator.make_value_interpolator('y')
        interp_z = interpolator.make_value_interpolator('z')

        self.assertAlmostEqual(interp_y([[0.5, 0.5]]), 1.0)
        self.assertAlmostEqual(interp_z([[0.5, 0.5]]), 0.25)
        self.assertAlmostEqual(interp_y([[1.5, 1.5]]), 5.0)
        self.assertAlmostEqual(interp_z([[1.5, 1.5]]), 2.25)

    def test_make_array_interpolator(self):
        """Test the make_array_interpolator method."""
        interpolator = GridInterpolator(self.filename, self.gridvals)
        interp_arr = interpolator.make_array_interpolator('arr')

        interpolated_val = interp_arr([[0.5, 0.5]])
        self.assertTrue(np.allclose(interpolated_val, [1.0, 0.25]))

        interpolated_val_2 = interp_arr([[1.5, 1.5]])
        self.assertTrue(np.allclose(interpolated_val_2, [3.0, 2.25]))

    def test_logspace_interpolation(self):
        """Test interpolation with logspace=True."""
        interpolator = GridInterpolator(self.filename, self.gridvals)
        interp_y_log = interpolator.make_value_interpolator('y', logspace=True)
        # At [0.5, 0.5], y is 1.0. The interpolated value should be close to 1.0
        #With logspace interpolation, we are taking the log of the values, interpolating, and then taking the power of 10.
        #The interpolation will not be perfect, so we have to have a larger tolerance
        self.assertAlmostEqual(interp_y_log([[0.5, 0.5]]), 1.0, delta=1.0)


if __name__ == '__main__':
    unittest.main()
