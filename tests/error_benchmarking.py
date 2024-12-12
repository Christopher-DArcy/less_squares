import unittest
import numpy as np
from less_squares import LessSquares

class TestLessSquaresOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up shared state for each test."""
        self.etol = 1e-16
        self.r_etol = 1e+3
        self.size_scalar = 200
        np.random.seed(42)
        self.iteration_count = 100
        self.results = []  # List to store subcase results

    def generate_matrix(self, shape):
        """Helper method to generate matrices of different shapes."""
        if shape == 'fat':
            m = 10 + int(np.random.random() * self.size_scalar)
            n = m + 10 + int(np.random.random() * self.size_scalar)
        elif shape == 'fat_flat':
            m = 1
            n = 10 + int(np.random.random() * self.size_scalar)
        elif shape == 'skinny':
            n = 10 + int(np.random.random() * self.size_scalar)
            m = n + 10 + int(np.random.random() * self.size_scalar)
        elif shape == 'skinny_flat':
            m = 10 + int(np.random.random() * self.size_scalar)
            n = 1
        elif shape == 'square':
            m = 10 + int(np.random.random() * self.size_scalar)
            n = m
        return np.random.random(size=(m, n)), m, n
    
    def test_add_operation(self):
        """Test the 'Add' operation."""
        for bias in [True, False]:
            for skew in [True, False]:
                for shape in ['square', 'skinny', 'fat']:
                    for index in [0, 3, -1]:
                        for direction in ['row', 'column']:
                            axis = 0 if direction == 'row' else 1
                            A, m, n = self.generate_matrix(shape)
                            if index < (axis * n + (1 - axis) * m):
                                model = LessSquares(A)
                                A_new = A.copy()
                                for k in range(self.iteration_count):
                                    u = self._generate_u(A, direction, bias, skew)
                                    model.add(u, index, axis)
                                    A_new = self._expected_add(A_new, u, index, direction)
                                self._assert_matrix(model.matrix, A_new, f'Add: shape fail shape:{shape}, axis:{direction}, index:{index}, bias:{bias}, skew:{skew}')
                                self._assert_pinv(model, shape, direction, index, bias, skew)
    
    def _generate_u(self, A, direction, bias, skew):
        """Helper function to generate u for testing."""
        if direction == 'row':
            u = np.random.random(size=(A.shape[1], 1)) if skew else np.random.normal(size=(A.shape[1], 1))
            if bias:
                u = A[0, :, np.newaxis] + 0.00001 * u
        else:
            u = np.random.random(size=(A.shape[0], 1)) if skew else np.random.normal(size=(A.shape[0], 1))
            if bias:
                u = A[:, 0, np.newaxis] + 0.00001 * u
        return u                

    def _assert_matrix(self, result, expected, message):
        """Helper function for matrix comparison."""
        self.assertTrue(np.allclose(result, expected), message)

    def _assert_pinv(self, model, shape, direction, index, bias, skew):
        """Check pseudo-inverse accuracy and record subcase results."""
        def full_check(A, A_p):
            c1 = A @ A_p @ A - A
            c2 = A_p @ A @ A_p - A_p
            c3 = (A @ A_p).T - (A @ A_p)
            c4 = (A_p @ A).T - A_p @ A
            return np.max(np.abs(c1)), np.max(np.abs(c2)), np.max(np.abs(c3)), np.max(np.abs(c4))
        
        if model.A.size > 0:
            package_error = max(full_check(model.A, model.pinv))
            numpy_error = max(full_check(model.A, np.linalg.pinv(model.A)))
            relative_error = package_error / numpy_error
            
            # Record the subcase results
            self.results.append({
                "shape": shape,
                "direction": direction,
                "index": index,
                "bias": bias,
                "skew": skew,
                "package_error": package_error,
                "numpy_error": numpy_error,
                "relative_error": relative_error
            })
            self.assertTrue(package_error < self.etol or relative_error < self.r_etol)
    
    def test_delete_operation(self):
        """Test the 'Delete' operation."""
        for shape in ['square', 'fat', 'skinny', 'fat_flat', 'skinny_flat']:
            for index in [0, 3, -1]:
                for direction in ['row', 'column']:
                    axis = 0 if direction == 'row' else 1
                    A, m, n = self.generate_matrix(shape)
                    model = LessSquares(A)
                    if ((m == 1 and axis == 0) or (n == 1 and axis == 1)) and index not in [0,1,-1]:
                        with self.assertRaises(IndexError):
                            model.delete(index, axis)
                    else:
                        model.delete(index, axis)
                        A_new = self._expected_delete(A, index, direction)
                        self._assert_matrix(model.matrix, A_new, f'Delete: shape fail {shape}, {direction}, {index}')
                        self._assert_pinv(model, f'Delete: pinv fail {shape}, {direction}, {index}')

    def test_append_operation(self):
        """Test the 'Append' operation."""
        for bias in [True,False]:
            for skew in [True,False]:
                for shape in ['square', 'fat', 'skinny', 'fat_flat', 'skinny_flat']:
                    for direction in ['row', 'column']:
                        axis = 0 if direction == 'row' else 1
                        A, m, n = self.generate_matrix(shape)
                        model = LessSquares(A)
                        u = self._generate_u(A, direction, bias, skew)
                        model.append(u, axis)
                        A_new = self._expected_append(A, u, direction)
                        self._assert_matrix(model.matrix, A_new, f'Append: shape fail shape:{shape}, axis:{direction}, bias:{bias}, skew:{skew}')
                        self._assert_pinv(model, f'Append: pinv fail shape:{shape}, axis:{direction}, bias:{bias}, skew:{skew}')

    def _expected_add(self, A, u, index, direction):
        """Return the expected result of the Add operation."""
        A_new = A.copy()
        if direction == 'row':
            A_new[index, :] += u.flatten()
        else:
            A_new[:, index] += u.flatten()
        return A_new

    def _expected_delete(self, A, index, direction):
        """Return the expected result of the Delete operation."""
        A_new = A.copy()
        if direction == 'row':
            A_new = np.delete(A_new, index, axis=0)
        else:
            A_new = np.delete(A_new, index, axis=1)
        return A_new

    def _expected_append(self, A, u, direction):
        """Return the expected result of the Append operation."""
        if direction == 'row':
            return np.vstack((A, u.T))
        else:
            return np.hstack((A, u))

    def benchmarking(self):
        """Print subcase results in a formatted table."""
        print(f"{'Shape':<10}{'Direction':<10}{'Index':<10}{'Bias':<10}{'Skew':<10}{'Pkg Error':<15}{'Numpy Error':<15}{'Rel Error':<15}")
        print("-" * 85)
        for result in self.results:
            print(f"{result['shape']:<10}{result['direction']:<10}{result['index']:<10}{str(result['bias']):<10}{str(result['skew']):<10}"
                  f"{result['package_error']:<15.5e}{result['numpy_error']:<15.5e}{result['relative_error']:<15.5e}")

# To run the tests and benchmark
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLessSquaresOperations)
    unittest.TextTestRunner().run(suite)
    test_case = TestLessSquaresOperations()
    test_case.setUp()
    test_case.test_add_operation()
    test_case.benchmarking()
