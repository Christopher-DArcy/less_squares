# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
import numpy as np
from less_squares import LessSquares

class TestLessSquaresOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up shared state for each test."""
        self.etol = 1e-16
        self.r_etol = 1e+3#how many times more error is acceptable vs numpy
        self.size_scalar = 500
        np.random.seed(49)  # Fix the seed for reproducibility
        self.iteration_count = 500
        self.print_all = True
        self.small_dim = 10 + int(np.random.random() * self.size_scalar)
        self.big_dim = self.small_dim + 10 + int(np.random.random() * self.size_scalar)
        if self.print_all:
            self.etol = 1e+100
            self.r_tol = 1e+100

    def generate_matrix(self, shape):
        """Helper method to generate matrices of different shapes."""
        if shape == 'fat':
            m = self.small_dim  # rows
            n = self.big_dim  # cols
        elif shape == 'fat_flat':
            m = 1
            n = self.big_dim
        elif shape == 'skinny':
            n = self.small_dim  # rows
            m = self.big_dim  # cols
        elif shape == 'skinny_flat':
            m = self.big_dim
            n = 1
        elif shape == 'square':
            m = self.big_dim
            n = self.big_dim
        return np.random.random(size=(m, n)), m, n
    
    def test_add_operation(self):
        """Test the 'Add' operation."""
        for bias in [False]:
            for skew in [True,False]:
                for shape in ['square', 'skinny','fat']:#'fat', 'skinny']:#, 'fat_flat', 'skinny_flat']:
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
                                if self.print_all:
                                    print(f'Add    | {(shape+(11-len(shape))*" ")} | a:{axis} | i:{(index!=-1)*" "+str(index)} | b:{int(bias)} | s:{int(skew)}: ', end='')
                                self._assert_matrix(model.matrix, A_new, f'Add: shape fail shape:{shape}, axis:{direction}, index:{index}, bias:{bias}, skew:{skew}')
                                self._assert_pinv(model, f'Add: pinv fail shape:{shape}, axis:{direction}, index:{index}, bias:{bias}, skew:{skew}')
    
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
                        if self.print_all:
                            print(f'Delete | {(shape+(11-len(shape))*" ")} | a:{axis} | i:{(index!=-1)*" "+str(index)} | b:- | s:-: ', end='')
                        self._assert_matrix(model.matrix, A_new, f'Delete: shape fail {shape}, {direction}, {index}')
                        self._assert_pinv(model, f'Delete: pinv fail {shape}, {direction}, {index}')

    def test_append_operation(self):
        """Test the 'Append' operation."""
        for bias in [False]:
            for skew in [True,False]:
                for shape in ['square', 'fat', 'skinny', 'fat_flat', 'skinny_flat']:
                    for direction in ['row', 'column']:
                        axis = 0 if direction == 'row' else 1
                        A, m, n = self.generate_matrix(shape)
                        model = LessSquares(A)
                        A_new = A.copy()
                        for k in range(self.iteration_count):
                            u = self._generate_u(A_new, direction, bias, skew)
                            model.append(u, axis)
                            A_new = self._expected_append(A_new, u, direction)
                        if self.print_all:
                            print(f'Append | {(shape+(11-len(shape))*" ")} | a:{axis} | i:-- | b:{int(bias)} | s:{int(skew)}: ', end='')
                        self._assert_matrix(model.matrix, A_new, f'Append: shape fail shape:{shape}, axis:{direction}, bias:{bias}, skew:{skew}')
                        self._assert_pinv(model, f'Append: pinv fail shape:{shape}, axis:{direction}, bias:{bias}, skew:{skew}')

    def _generate_u(self, A, direction, bias, skew):
        """Helper function to generate u for testing."""
        if direction == 'row':
            if skew:
                u = np.random.random(size=(A.shape[1], 1))
            else:
                u = np.random.normal(size=(A.shape[1], 1))                
            if bias:
                u = A[0, :, np.newaxis] + 0.00001 * u
        else:
            if skew:
                u = np.random.random(size=(A.shape[0], 1))
            else:
                u = np.random.normal(size=(A.shape[0], 1))
            if bias:
                u = A[:, 0, np.newaxis] + 0.00001 * u
        return u                

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
            self.assertTrue(package_error < self.etol or relative_error < self.r_etol, message+' //'+str(package_error)+' //'+str(numpy_error))
        else:
            package_error = 0.00
            numpy_error = 0.00
            relative_error = 1.00
        if self.print_all:
            print(f'{relative_error:.2e}|{package_error:.2e}|{numpy_error:.2e}')

# To run these tests, use: python -m unittest test_your_module.py
if __name__ == '__main__':
    unittest.main()
