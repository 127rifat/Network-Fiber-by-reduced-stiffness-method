# fiber_network.py

import numpy as np
from scipy.sparse import csr_matrix
import scipy.io as scio
from scipy.io import mmwrite
from scipy.sparse.linalg import inv 

class Node(object):
    def __init__(self, num, x, y, z, dof1, dof2, dof3):
        self.num = num
        self.coords = [x, y, z]
        self.dof_num = [dof1, dof2, dof3]

class Element(object):
    def __init__(self, n1, n2, youngs_modulus, fiber_area, n1_deformed, n2_deformed):
        self.node1 = n1
        self.node2 = n2
        self.node1_deformed = n1_deformed
        self.node2_deformed = n2_deformed
        self.fiber_area = fiber_area
        self.youngs_modulus = youngs_modulus

    def get_length(self):
        return np.linalg.norm(np.array(self.node2.coords) - np.array(self.node1.coords))

    def get_length_deformed(self):
        return np.linalg.norm(np.array(self.node2_deformed.coords) - np.array(self.node1_deformed.coords))

    def get_unit_vector_deformed(self):
        return (np.array(self.node2_deformed.coords) - np.array(self.node1_deformed.coords)) / self.get_length_deformed()

    def get_k(self):
        L = self.get_length_deformed()
        L0 = self.get_length()
        stretch = L / L0
        P = self.compute_PK1_stress(self.youngs_modulus, self.fiber_area, stretch)
        n = self.get_unit_vector_deformed()
        LVec = L * n
        LNorm = np.linalg.norm(LVec)
        I = np.identity(3)
        dpdl = (self.youngs_modulus * self.fiber_area / 2 * (3 * L**2 / L0**3 - 1 / L0)) * n
        dfdu = np.outer(dpdl, n) + (1 / LNorm * I - np.outer((1 / LNorm**3) * LVec, LVec)) * P
        ke = np.zeros((6, 6))
        ke[0:3, 0:3] = dfdu
        ke[0:3, 3:6] = -dfdu
        ke[3:6, 0:3] = -dfdu
        ke[3:6, 3:6] = dfdu
        return ke

    @staticmethod
    def compute_PK1_stress(E, A0, stretch):
        return E * (stretch**2 - 1) / 2 * A0 * stretch

class FiberNetworkParams(object):
    def __init__(self, E, R):
        self.youngs_modulus = E
        self.fiber_radius = R

def read_network_params(filename):
    params = []
    param_assignment = []
    with open(filename, 'r') as f:
        line1 = f.readline()
        num_params, _ = [int(x) for x in line1.split()]
        for _ in range(num_params):
            line = f.readline()
            _, R, E = [float(x) for x in line.split()]
            params.append(FiberNetworkParams(E, R))
        param_assignment = np.loadtxt(filename, skiprows=num_params + 1, dtype=int).tolist()
    return params, param_assignment

def read_network(filename, deformed_filename, params, param_assignment):
    with open(filename, 'r') as f:
        num_nodes, num_fibers, _ = [int(x) for x in f.readline().split()]
        nodes = [Node(i, *map(float, f.readline().split()), dof, dof + 1, dof + 2) for i, dof in enumerate(range(0, 3 * num_nodes, 3))]

    with open(deformed_filename, 'r') as f:
        assert [int(x) for x in f.readline().split()][:2] == [num_nodes, num_fibers]
        nodes_deformed = [Node(i, *map(float, f.readline().split()), dof, dof + 1, dof + 2) for i, dof in enumerate(range(0, 3 * num_nodes, 3))]

    elements = []
    with open(filename, 'r') as f:
        for _ in range(num_nodes + 1):
            next(f)
        for _ in range(num_fibers):
            line = f.readline().split()
            n1, n2 = int(line[0]), int(line[1])
            elements.append(Element(nodes[n1], nodes[n2], params[param_assignment[_]].youngs_modulus, np.pi * params[param_assignment[_]].fiber_radius**2, nodes_deformed[n1], nodes_deformed[n2]))

    return nodes, elements

class FiberNetwork(object):
    def __init__(self, filename_undeformed, filename_deformed, param_filename):
        self.params, self.param_assignment = read_network_params(param_filename)
        self.nodes, self.elements = read_network(filename_undeformed, filename_deformed, self.params, self.param_assignment)
        self.dof_max = self.nodes[-1].dof_num[-1] + 1
        self.K = np.zeros((self.dof_max, self.dof_max))

    def construct_global_stiffness(self):
        for element in self.elements:
            ke = element.get_k()
            assert np.allclose(ke, ke.T)
            global_dofs = element.node1.dof_num + element.node2.dof_num
            for i in range(6):
                for j in range(6):
                    self.K[global_dofs[i], global_dofs[j]] += ke[i, j]
        self.K = csr_matrix(self.K)
        self.K = 0.5 * (self.K + self.K.T)
        return self.K

    def compute_reduced_stiffness(self, n_boundary_dofs):
        K_BB = self.K[:n_boundary_dofs, :n_boundary_dofs]
        K_BI = self.K[:n_boundary_dofs, n_boundary_dofs:]
        K_IB = self.K[n_boundary_dofs:, :n_boundary_dofs]
        K_II = self.K[n_boundary_dofs:, n_boundary_dofs:]
        K_II_inv = inv(K_II)
        K_red = K_BB - K_BI.dot(K_II_inv.dot(K_IB))
        K_red = 0.5*(K_red + K_red.transpose())
        return K_red

def count_boundary_nodes(undeformed_mesh_file, boundary_value=0.5, rtol=1e-5, atol=1e-8):
    with open(undeformed_mesh_file, 'r') as file:
        lines = file.readlines()
    num_boundary_nodes = 0
    for line in lines[1:]:
        x, y, z = map(float, line.split()[:3])
        if (np.isclose(abs(x), boundary_value, rtol=rtol, atol=atol) or
            np.isclose(abs(y), boundary_value, rtol=rtol, atol=atol) or
            np.isclose(abs(z), boundary_value, rtol=rtol, atol=atol)):
            num_boundary_nodes += 1
        else:
            # Stop the count when the first non-boundary node is encountered
            break
    return num_boundary_nodes

def process_fiber_network(undeformed_mesh_file, deformed_mesh_file, mesh_params_file):
    fn = FiberNetwork(undeformed_mesh_file, deformed_mesh_file, mesh_params_file)
    fn.construct_global_stiffness()
    n_boundary_nodes = count_boundary_nodes(undeformed_mesh_file)
    n_boundary_dofs = n_boundary_nodes * 3
    K_red = fn.compute_reduced_stiffness(n_boundary_dofs)
    output_matrix_file = 'Global_reduced_stiffness_matrix.mtx'
    mmwrite(output_matrix_file, K_red)
    return output_matrix_file  # Optionally return the file path

# The following code block is to prevent the script from executing when imported as a module
if __name__ == "__main__":
    undeformed_mesh_file = 'rearranged_mesh_tr.txt'  # Placeholder, replace with actual path
    deformed_mesh_file = 'deformed_co_for_reduced.txt'  # Placeholder, replace with actual path
    mesh_params_file = 'rearranged_mesh_tr.txt.params'  # Placeholder, replace with actual path
    
    # This will execute the process when the script is run standalone
    output_matrix_file = process_fiber_network(undeformed_mesh_file, deformed_mesh_file, mesh_params_file)
    print(f"Reduced stiffness matrix has been saved to: {output_matrix_file}")

