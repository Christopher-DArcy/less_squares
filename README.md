# less_squares

The less_squares object wraps a numpy matrix and will dynamically update the matrix pseudoinverse with O(nm) cost when rows are are modified in the underlying matrix. This allows for regression operations to be performed at low cost on regularly updated data-sets.

## Features

- Compute and update pseudoinverses dynamically.
- Add or remove rows/columns without recomputing the entire pseudoinverse.
- Swap rows/columns or update their values.
- Check the validity of the pseudoinverse with various modes.
- Expand matrices with placeholders for future updates.

## Installation

Install the package via pip:
```
pip install less_squares
```
## Usage

### Importing and Initialization
```
from less_squares import LessSquares
import numpy as np

# Initialize with a matrix
matrix = np.random.rand(5, 3)
ls = LessSquares(matrix)
```
### Properties
- Retrieve the pseudoinverse:
```
pseudoinverse = ls.pseudo
```
- Retrieve the original matrix:
```
original_matrix = ls.matrix
```
### Methods

#### Adding and Removing Rows/Columns
- Add a vector:
```
vector = np.random.rand(matrix.shape[1])
ls.append(vector, axis=0)  # Add as a new row
```
- Remove a row or column:
```
ls.delete(index=1, axis=0)  # Delete the second row
```

#### Updating Values
- Update a row or column:
```
updated_vector = np.random.rand(matrix.shape[1])
ls.update(updated_vector, index=2, axis=0)  # Update the third row
```
#### Swapping Rows/Columns
- Swap two rows or columns:
```
ls.swap_slices(axis=0, index1=0, index2=2)  # Swap the first and third rows
```

#### Expanding the Matrix
- Add a placeholder row or column:
```
ls.expand(axis=0)  # Add a placeholder row
```
#### Validating the Pseudoinverse
- Check validity:
```
is_valid = ls.check(mode='fast')  # Quick validity check
max_errors = ls.check(mode='full')  # Detailed error metrics
```
### Example
```
# Initialize the class with a random matrix
matrix = np.random.rand(4, 3)
ls = LessSquares(matrix)

# Add a new row
new_row = np.random.rand(matrix.shape[1])
ls.append(new_row, axis=0)

# Update a column
new_column = np.random.rand(matrix.shape[0])
ls.update(new_column, index=1, axis=1)

# Check the pseudoinverse
print("Pseudoinverse is valid:", not ls.check(mode='fast'))
```

## Requirements
- Python 3.8+
- NumPy

## License
This project is licensed under the GNU GPL3 License.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub.

Fixes for issues with ill-conditioned matrices are in the works, the idea will be to store linearly dependent collumns seperately from the rest of the matrix itself and use the projection matrix to decide if new collumns should be allocated into that.

## Author
Christopher D'Arcy
