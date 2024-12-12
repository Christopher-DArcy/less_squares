# -*- coding: utf-8 -*-

import unittest
import numpy as np
from less_squares import LessSquares

class TestLessSquaresOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up shared state for each test."""
        self.etol = 1e-16
        self.r_etol = 1e+1#how many times more error is acceptable vs numpy
        self.size_scalar = 200
        np.random.seed(42)  # Fix the seed for reproducibility

    def generate_matrix(self, shape):
        """Helper method to generate matrices of different shapes."""
        if shape == 'fat':
            m = 10 + int(np.random.random() * self.size_scalar)  # rows
            n = m + 10 + int(np.random.random() * self.size_scalar)  # cols
        elif shape == 'fat_flat':
            m = 1
            n = 10 + int(np.random.random() * self.size_scalar)
        elif shape == 'skinny':
            n = 10 + int(np.random.random() * self.size_scalar)  # rows
            m = n + 10 + int(np.random.random() * self.size_scalar)  # cols
        elif shape == 'skinny_flat':
            m = 10 + int(np.random.random() * self.size_scalar)
            n = 1
        elif shape == 'square':
            m = 10 + int(np.random.random() * self.size_scalar)
            n = m
        return np.random.random(size=(m, n)), m, n
    
    def test_add_operation(self):
        """Test the 'Add' operation."""
        for shape in ['square', 'fat', 'skinny', 'fat_flat', 'skinny_flat']:
            for index in [0, 3, -1]:
                for direction in ['row', 'column']:
                    axis = 0 if direction == 'row' else 1
                    A, m, n = self.generate_matrix(shape)
                    if index < (axis * n + (1 - axis) * m):
                        model = LessSquares(A)
                        u = self._generate_u(A, direction)
                        model.add(u, index, axis)
                        A_new = self._expected_add(A, u, index, direction)
                        self._assert_matrix(model.matrix, A_new, f'Add: shape fail {shape}, {direction}, {index}')
                        self._assert_pinv(model, f'Add: pinv fail {shape}, {direction}, {index}')
    
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
        for shape in ['square', 'fat', 'skinny', 'fat_flat', 'skinny_flat']:
            for direction in ['row', 'column']:
                axis = 0 if direction == 'row' else 1
                A, m, n = self.generate_matrix(shape)
                model = LessSquares(A)
                u = self._generate_u(A, direction)
                model.append(u, axis)
                A_new = self._expected_append(A, u, direction)
                self._assert_matrix(model.matrix, A_new, f'Append: shape fail {shape}, {direction}')
                self._assert_pinv(model, f'Append: pinv fail {shape}, {direction}')

    def _generate_u(self, A, direction):
        """Helper function to generate u for testing."""
        if direction == 'row':
            return A[0, :, np.newaxis] + 0.00001 * np.random.random(size=(A.shape[1], 1))
        else:
            return A[:, 0, np.newaxis] + 0.00001 * np.random.random(size=(A.shape[0], 1))

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

    def _assert_matrix(self, result, expected, message):
        """Helper function for matrix comparison."""
        self.assertTrue(np.allclose(result, expected), message)

    def _assert_pinv(self, model, message):
        def full_check(A,A_p):
            c1 = A @ A_p @ A - A
            c2 = A_p @ A @ A_p - A_p
            c3 = (A @ A_p).T - (A @ A_p)
            c4 = (A_p @ A).T - A_p @ A
            return np.max(np.abs(c1)),np.max(np.abs(c2)),np.max(np.abs(c3)),np.max(np.abs(c4))
        """Helper function to check pseudo-inverse accuracy."""
        if model.A.size > 0:
            package_error = max(full_check(model.A, model.pinv))
            numpy_error = max(full_check(model.A, np.linalg.pinv(model.A)))
            relative_error = package_error/numpy_error
            self.assertTrue(package_error < self.etol or relative_error < self.r_etol, message+str(package_error)+' '+str(numpy_error))

# To run these tests, use: python -m unittest test_your_module.py
if __name__ == '__main__':
    unittest.main()
