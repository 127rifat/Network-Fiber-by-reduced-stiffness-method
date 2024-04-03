# tensor_operations.py
import numpy as np
from numpy import linalg as LA
from scipy.linalg import polar
import math as mt

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def to_mandel(M):
    """
    Convert a fourth-order tensor with minor symmetries to its Mandel representation.
    """
    
    M_mandel = np.zeros((6, 6))
    index_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    
    for i, (i1, i2) in enumerate(index_map):
        for j, (j1, j2) in enumerate(index_map):
            if i < 3 and j < 3:  
                M_mandel[i, j] = M[i1, i2, j1, j2]
            elif i >= 3 and j >= 3:  
                M_mandel[i, j] = 2 * M[i1, i2, j1, j2]
            else:  
                M_mandel[i, j] = np.sqrt(2) * M[i1, i2, j1, j2]
                
    return M_mandel

def from_mandel(Mandel_inv):
    """
    Convert the Mandel representation back to the original fourth order tensor form,
    ensuring that minor symmetries are correctly restored.
    """
    M_out = np.zeros((3, 3, 3, 3))
    index_map = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    for i, (i1, i2) in enumerate(index_map):
        for j, (j1, j2) in enumerate(index_map):
            # Determine the scaling factor
            factor = 2 if i >= 3 and j >= 3 else np.sqrt(2) if i >= 3 or j >= 3 else 1

            # Assign values with appropriate scaling, considering minor symmetries
            value = Mandel_inv[i, j] / factor
            M_out[i1, i2, j1, j2] = value
            M_out[i2, i1, j1, j2] = value
            M_out[i1, i2, j2, j1] = value
            M_out[i2, i1, j2, j1] = value

    return M_out

if __name__ == "__main__":
    