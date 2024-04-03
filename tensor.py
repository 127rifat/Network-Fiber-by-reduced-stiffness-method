# tensor_computation.py

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

def compute_tensor(stiffness_matrix_file, boundary_nodes_file, n_nodes, n_dof, V0):
    K_sparse = mmread(stiffness_matrix_file)
    K = csr_matrix(K_sparse)
    X = np.loadtxt(boundary_nodes_file)
    C_M = np.zeros((n_dof, n_dof, n_dof, n_dof))

    for a in range(n_dof):
        for b in range(n_dof):
            for c in range(n_dof):
                for d in range(n_dof):
                    sum_ij = 0
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            K_index_i = i * n_dof + a
                            K_index_j = j * n_dof + c
                            K_value = K[K_index_i, K_index_j]
                            sum_ij += K_value * X[i, b] * X[j, d]
                    C_M[a, b, c, d] = sum_ij / V0

    return C_M

def save_tensor(C_M, output_file_path, tensor_file_path, n_dof):
    with open(output_file_path, 'w') as f:
        for a in range(n_dof):
            for b in range(n_dof):
                for c in range(n_dof):
                    for d in range(n_dof):
                        f.write(f'C_M[{a+1},{b+1},{c+1},{d+1}] = {C_M[a, b, c, d]:.4e}\n')

    np.save(tensor_file_path, C_M)
